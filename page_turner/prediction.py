import cv2
import pyaudio
import torch
import threading
import wave

import matplotlib.cm as cm
import numpy as np

from collections import Counter
from cyolo_score_following.models.yolo import load_pretrained_model
from cyolo_score_following.utils.data_utils import SAMPLE_RATE, FPS, FRAME_SIZE, HOP_SIZE
from cyolo_score_following.utils.general import xywh2xyxy
from cyolo_score_following.utils.video_utils import plot_box, plot_line
from cyolo_score_following.utils.general import load_wav
from multiprocessing import Pipe
from page_turner.camera import Camera
from page_turner.config import *
from scipy import interpolate


class ScoreAudioPrediction(threading.Thread):
    def __init__(self, param_path, live_audio=None, audio_path=None, live_score=None, score_path=None,
                 gt_only=False, page=None):
        """
        This function initializes an instance of the class.
        :param param_path: full path to the model loaded
        :param live_audio:
        :param audio_path: None or full path to the .wav file of the audio chosen
        :param live_score:
        :param score_path: None or full path to the .npz file of the score chosen
        :param gt_only: whether to also show the prediction or only the ground truth
        :param page:
        """
        threading.Thread.__init__(self)

        self.gt_only = gt_only
        self.live_audio = live_audio
        self.audio_path = audio_path
        self.live_score = live_score
        self.score_path = score_path
        self.org_scores, self.score, self.systems, self.interpol_fnc, \
            self.pad, self.scale_factor, self.n_pages = [None] * 7

        self.camera = None
        self.p_output, self.p_input = None, None
        self.audio_stream, self.wave_file = None, None
        self.pa = None
        self.score_img, self.spec_img = None, None

        # load all essential variables from files or for live
        self.load_essentials()
        # load the plots of the pages for the score page overview
        self.page_plots = self.load_page_plots()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network, criterion = load_pretrained_model(param_path)

        print(self.network)
        print("Putting model to %s ..." % self.device)
        self.network.to(self.device)
        print("Number of parameters:", sum(p.numel() for p in self.network.parameters() if p.requires_grad))
        self.network.eval()

        self.actual_page = 0
        self.track_page = page
        self.start_ = None
        self.vis_spec = None
        self.is_piece_end = False

    def load_essentials(self):
        """
        This function loads all essential variables for running.
        Either from .wav files or live audio and either from .npz files
        or live score images.
        :return:
        """
        self.pa = pyaudio.PyAudio()
        if self.audio_path is not None:
            # read from file
            self.wave_file = wave.open(self.audio_path, 'rb')
            self.audio_stream = self.pa.open(format=self.pa.get_format_from_width(self.wave_file.getsampwidth()),
                                             channels=self.wave_file.getnchannels(),
                                             rate=self.wave_file.getframerate(),
                                             output=True)
        else:
            # live audio input
            self.audio_stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                                             input=True, frames_per_buffer=FRAME_SIZE // 2)

        if self.score_path is not None:
            self.org_scores, self.score, self.systems, self.interpol_fnc, self.pad, self.scale_factor, self.n_pages = \
                self.load_score()
        else:
            self.p_output, self.p_input = Pipe()
            self.camera = Camera(pipe=self.p_output)
            self.camera.start()

            self.pad = PADDING
            self.scale_factor = SCALE_FACTOR

            org_score, scaled_score = self.p_input.recv()
            self.org_scores = np.asarray([org_score])
            self.score = np.asarray([scaled_score])

    def load_score(self):
        """
        This function loads all variables required from the score.
        :return:
        """
        npzfile = np.load(self.score_path, allow_pickle=True)

        org_scores = npzfile["sheets"]
        coords, systems = list(npzfile["coords"]), list(npzfile['systems'])

        n_pages, h, w = org_scores.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

        # Determine padding
        pad = ((0, 0), (0, 0), (pad1, pad2))

        # Add padding
        padded_scores = np.pad(org_scores, pad, mode="constant", constant_values=255)

        onsets = []
        for i in range(len(coords)):
            coords[i]['note_x'] += pad1

            # onset time to frame
            coords[i]['onset'] = int(coords[i]['onset'] * FPS)
            onsets.append(coords[i]['onset'])

        for i in range(len(systems)):
            systems[i]['x'] += pad1

        onsets = np.asarray(onsets, dtype=np.int)

        onsets = np.unique(onsets)
        coords_new = []
        for onset in onsets:
            onset_coords = list(filter(lambda x: x['onset'] == onset, coords))

            onset_coords_merged = {}
            for entry in onset_coords:
                for key in entry:
                    if key not in onset_coords_merged:
                        onset_coords_merged[key] = []
                    onset_coords_merged[key].append(entry[key])

            # get system and page with most notes in it
            system_idx = int(Counter(onset_coords_merged['system_idx']).most_common(1)[0][0])
            note_x = np.mean(
                np.asarray(onset_coords_merged['note_x'])[np.asarray(onset_coords_merged['system_idx']) == system_idx])
            page_nr = int(Counter(onset_coords_merged['page_nr']).most_common(1)[0][0])

            # set y to staff center
            note_y = systems[system_idx]['y']
            coords_new.append([note_y, note_x, system_idx, page_nr])

        coords_new = np.asarray(coords_new)

        # we want to match the frames to the coords of the previous onset, as the notes at the next coord position
        # aren't played yet
        interpol_fnc = interpolate.interp1d(onsets, coords_new.T, kind='previous', bounds_error=False,
                                            fill_value=(coords_new[0, :], coords_new[-1, :]))

        scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

        # scale scores
        scaled_score = []
        scale_factor = scores[0].shape[0] / SCALE_WIDTH

        for score in scores:
            scaled_score.append(cv2.resize(score, (SCALE_WIDTH, SCALE_WIDTH), interpolation=cv2.INTER_AREA))

        score = np.stack(scaled_score)

        org_scores_rgb = []
        for org_score in org_scores:
            org_score = np.array(org_score, dtype=np.float32) / 255.

            org_scores_rgb.append(cv2.cvtColor(org_score, cv2.COLOR_GRAY2BGR))

        return org_scores_rgb, score, systems, interpol_fnc, pad1, scale_factor, n_pages

    def load_audio(self):
        """
        This function loads all variables required from the audio.
        :return:
        """
        signal = load_wav(self.audio_path, sr=SAMPLE_RATE)
        return signal

    def load_page_plots(self):
        """
        This function loads the plots for all score pages,
        which will be shown in a grid in the main application.
        :return: page_plots: numpy arrays for each score page
        """
        page_plots = []

        if self.n_pages is not None:
            for curr_page in range(self.n_pages):
                img_pred = cv2.cvtColor(self.org_scores[curr_page], cv2.COLOR_RGB2BGR)
                page_plots.append(np.array((img_pred * 255), dtype=np.uint8))

        return page_plots

    def get_next_images(self):
        return self.score_img, self.spec_img

    def camera_input(self):
        return self.camera is not None

    def run(self):
        self.is_piece_end = False
        signal = None

        from_ = 0
        to_ = FRAME_SIZE

        score_tensor = torch.from_numpy(self.score).unsqueeze(1).to(self.device)

        hidden = None
        frame_idx = 0

        while not self.is_piece_end:

            if self.wave_file is None:
                data = self.audio_stream.read(FRAME_SIZE // 2)
            else:
                data = self.wave_file.readframes(FRAME_SIZE // 2)

            if len(data) <= 0:
                break

            if self.wave_file is not None:
                self.audio_stream.write(data)

            sig_excerpt = np.frombuffer(data, dtype=np.int16) / 2 ** 15
            signal = np.concatenate((signal, sig_excerpt)) if signal is not None else sig_excerpt

            if len(signal[from_:to_]) != FRAME_SIZE:
                continue

            if self.camera_input():
                self.actual_page = 0
                self.track_page = 0

                if self.p_input.poll():
                    org_score, scaled_score = self.p_input.recv()
                    self.org_scores = np.asarray([org_score])
                    self.score = np.asarray([scaled_score])

                score_tensor = torch.from_numpy(self.score).unsqueeze(0).to(self.device)

            else:
                true_position = np.array(self.interpol_fnc(frame_idx), dtype=np.float32)

                if self.actual_page != int(true_position[-1]):
                    hidden = None

                self.actual_page = int(true_position[-1])
                system = self.systems[int(true_position[2])]
                true_position = true_position[:2]

            if self.track_page is None or self.actual_page == self.track_page:
                self.start_ = from_ if self.start_ is None else self.start_

                with torch.no_grad():
                    sig_excerpt = torch.from_numpy(signal[from_:to_]).float().to(self.device)
                    spec_frame = self.network.compute_spec([sig_excerpt], tempo_aug=False)[0]

                    z, hidden = self.network.conditioning_network.get_conditioning(spec_frame, hidden=hidden)
                    inference_out, pred = self.network.predict(score_tensor[self.actual_page:self.actual_page + 1], z)

                _, idx = torch.sort(inference_out[0, :, 4], descending=True)
                filtered_pred = inference_out[0, idx[:1]]
                box = filtered_pred[..., :4]
                conf = filtered_pred[..., 4]
                x1, y1, x2, y2 = xywh2xyxy(box).cpu().numpy().T

                x1 = x1 * self.scale_factor - self.pad
                x2 = x2 * self.scale_factor - self.pad
                y1 = y1 * self.scale_factor
                y2 = y2 * self.scale_factor

                if self.vis_spec is not None:
                    self.vis_spec = np.roll(self.vis_spec, -1, axis=1)
                else:
                    self.vis_spec = np.zeros((spec_frame.shape[-1], 400))

                self.vis_spec[:, -1] = spec_frame[0].cpu().numpy()

                # that is what i need for the app
                img_pred = cv2.cvtColor(self.org_scores[self.actual_page], cv2.COLOR_RGB2BGR)

                if not self.camera_input():

                    height = system['h'] / 2
                    center_y, center_x = true_position

                    plot_line([center_x - self.pad, center_y, height], img_pred, label="GT",
                              color=(0.96, 0.63, 0.25), line_thickness=2)

                if not self.gt_only:
                    plot_box([x1, y1, x2, y2], img_pred, label="Pred", color=(0, 0, 1), line_thickness=2)

                self.score_img = np.array((img_pred * 255), dtype=np.uint8)
                spec_excerpt = cv2.resize(np.flipud(self.vis_spec), (round(self.vis_spec.shape[1] * self.scale_factor),
                                                                     round(self.vis_spec.shape[0] * self.scale_factor)))
                spec_excerpt = cm.viridis(spec_excerpt)[:, :, :3]
                self.spec_img = np.array((spec_excerpt * 255), dtype=np.uint8)

            else:
                if self.start_ is not None:
                    # avoid moving back to the page
                    # (in case repetitions span across multiple pages, shouldn't happen in msmd)
                    self.is_piece_end = True

            from_ += HOP_SIZE
            to_ += HOP_SIZE
            frame_idx += 1

        if self.wave_file is not None:
            self.wave_file.close()

        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.pa.terminate()

        self.is_piece_end = True

    def stop_playing(self):
        self.is_piece_end = True