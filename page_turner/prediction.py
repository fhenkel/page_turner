import copy
import cv2
import serial
import torch
import threading

import matplotlib.cm as cm
import numpy as np

from collections import deque
from cyolo_score_following.models.yolo import load_pretrained_model
from cyolo_score_following.utils.data_utils import SAMPLE_RATE, FPS, FRAME_SIZE, HOP_SIZE
from cyolo_score_following.utils.general import xywh2xyxy
from cyolo_score_following.utils.video_utils import plot_box, create_video

from page_turner.audio_stream import AudioStream
from page_turner.config import *
from page_turner.image_stream import ImageStream


class ScoreAudioPrediction(threading.Thread):

    def __init__(self, param_path,  audio_path=None, score_path=None, n_pages=None, score_fraction=0.5):
        """
        This function initializes an instance of the class.
        :param param_path: full path to the model loaded
        :param audio_path: None or full path to the .wav file of the audio chosen
        :param score_path: None or full path to the .npz file of the score chosen
        :param n_pages: Number of pages to track if the score is coming from a camera input
        """
        threading.Thread.__init__(self)

        self.audio_path = audio_path
        self.score_path = score_path
        self.org_scores, self.score, self.pad, self.scale_factor = [None] * 4

        self.score_img, self.spec_img = None, None
        self.previous_prediction = None

        self.audio_stream = AudioStream(self.audio_path)
        self.image_stream = ImageStream(self.score_path, n_pages)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network, criterion = load_pretrained_model(param_path)

        print(f"Putting model to {self.device} ...")
        self.network.to(self.device)
        self.network.eval()

        self.actual_page = 0
        self.start_ = None
        self.vis_spec = None
        self.is_piece_end = False
        self.score_fraction = score_fraction

    def get_next_images(self):
        return self.score_img, self.spec_img

    def get_best_prediction(self, predictions, systems, start_from_top=False):

        _, idx = torch.sort(predictions[:, 4], descending=True)
        sorted_predictions = predictions[idx]

        sorted_predictions[:, :4] *= self.image_stream.scale_factor
        sorted_predictions[:, 0] -= self.image_stream.pad

        confidence = sorted_predictions[:, 4].cpu().numpy()

        best = self.previous_prediction

        x, y = sorted_predictions[:, :2].cpu().numpy().T
        x1, y1, x2, y2 = xywh2xyxy(sorted_predictions[:, :4]).cpu().numpy().T

        try:

            in_first_system = (y >= systems[0][0]) & (y <= systems[0][1])
            start_in_front = (x < SCORE_WIDTH*0.3)

            if start_from_top:
                indices = in_first_system & start_in_front & start_from_top

                if any(indices):
                    x1 = x1[indices]
                    x2 = x2[indices]
                    y1 = y1[indices]
                    y2 = y2[indices]
                    best = [x1[0], y1[0], x2[0], y2[0]]
            else:

                previous_y = self.previous_prediction[1] \
                                  + (self.previous_prediction[3] - self.previous_prediction[1])/2

                curr_system_idx = np.argwhere((systems[:, 0] <= previous_y) & (systems[:, 1] >= previous_y)).item()

                prev_system_idx = max(0, curr_system_idx - 1)
                next_system_idx = min(curr_system_idx + 1, len(systems) - 1)

                curr_system = systems[curr_system_idx]
                prev_system = systems[prev_system_idx]
                next_system = systems[next_system_idx]

                stay_within_system = (y >= curr_system[0]) & (y <= curr_system[1])
                move_to_prev_system = (y >= prev_system[0]) & (y <= prev_system[1]) & (confidence > 0.5)
                move_to_next_system = (y >= next_system[0]) & (y <= next_system[1])

                indices = stay_within_system | move_to_next_system | move_to_prev_system

                if any(indices):
                    x1 = x1[indices]
                    x2 = x2[indices]
                    y1 = y1[indices]
                    y2 = y2[indices]
                    best = [x1[0], y1[0], x2[0], y2[0]]

        except ValueError:
            # Fall back solution in case of an error (e.g. significant change in the detected systems)
            best = [x1[0], y1[0], x2[0], y2[0]]

        return best

    def run(self):
        self.is_piece_end = False
        signal = None

        from_ = 0
        to_ = FRAME_SIZE

        org_score, score, system_ys = self.image_stream.get(self.actual_page)

        hidden = None
        frame_idx = 0

        th_len = 5

        curr_y = deque(np.zeros(th_len), maxlen=th_len)
        curr_x = deque(np.zeros(th_len), maxlen=th_len)
        observations = []

        self.previous_prediction = None
        page_turner_cooldown = 0
        start_from_top = True
        while not self.is_piece_end:

            frame = self.audio_stream.get()

            if frame is None:
                break

            signal = np.concatenate((signal, frame)) if signal is not None else frame

            if len(signal[from_:to_]) != FRAME_SIZE:
                continue

            in_last_system = len(system_ys) > 0 and system_ys[-1][0] <= np.mean(curr_y) <= system_ys[-1][1]

            # if the current position in the last system reaches a certain threshold
            if in_last_system and np.mean(curr_x) > self.score_fraction * SCORE_WIDTH:

                if self.image_stream.more_pages(self.actual_page) and page_turner_cooldown <= 0:
                    print('Turn page')
                    hidden = None
                    self.actual_page += 1
                    self.previous_prediction = None

                    curr_y = deque(np.zeros(th_len), maxlen=th_len)
                    curr_x = deque(np.zeros(th_len), maxlen=th_len)
                    page_turner_cooldown = 40
                    start_from_top = True

                    if self.image_stream.camera_input:
                        try:
                            with serial.Serial('/dev/ttyUSB0', 9600, timeout=1) as ser:
                                ser.write(serial.to_bytes([0xA0, 0x01, 0x01, 0xA2]))
                                ser.write(serial.to_bytes([0xA0, 0x01, 0x00, 0xA1]))
                        except:
                            print("Physical page turning not possible. Did you run sudo chown <user> /dev/ttyUSB0 ?")

            org_score, score, system_ys = self.image_stream.get(self.actual_page)

            if page_turner_cooldown > 0:
                page_turner_cooldown -= 1
                hidden = None
                self.previous_prediction = None
                start_from_top = True

            with torch.no_grad():
                # add channel and batch dimension
                score_tensor = torch.from_numpy(score).unsqueeze(0).unsqueeze(0).to(self.device)

                sig_excerpt = torch.from_numpy(signal[from_:to_]).float().to(self.device)
                spec_frame = self.network.compute_spec([sig_excerpt], tempo_aug=False)[0]

                z, hidden = self.network.conditioning_network.get_conditioning(spec_frame, hidden=hidden)
                inference_out, pred = self.network.predict(score_tensor, z)

            filtered_inference_out = inference_out[0, inference_out[0, :, -1] == 0]

            best_prediction = self.get_best_prediction(filtered_inference_out, system_ys, start_from_top=start_from_top)

            if best_prediction is not None:
                x1, y1, x2, y2 = best_prediction

                curr_y.append(y1 + (y2 - y1) / 2)
                curr_x.append(x1 + (x2 - x1) / 2)

            start_from_top = False

            self.previous_prediction = best_prediction

            self.score_img, self.spec_img = self.prepare_visualization(org_score, spec_frame, best_prediction)

            observations.append(self.score_img)

            from_ += HOP_SIZE
            to_ += HOP_SIZE
            frame_idx += 1

        self.write_video(observations, signal)

        self.audio_stream.close()
        self.image_stream.close()

        self.is_piece_end = True

    def prepare_visualization(self, score, spec_frame, prediction):

        if self.vis_spec is not None:
            self.vis_spec = np.roll(self.vis_spec, -1, axis=1)
        else:
            self.vis_spec = np.zeros((spec_frame.shape[-1], SPEC_VIS_WINDOW))

        self.vis_spec[:, -1] = spec_frame[0].cpu().numpy()

        img_pred = copy.copy(score)

        if prediction is not None:
            x1, y1, x2, y2 = prediction
            plot_box([x1, y1, x2, y2], img_pred, label="Pred", color=(1, 0, 0), line_thickness=2)

        score_img = np.array((img_pred * 255), dtype=np.uint8)

        spec_excerpt = cv2.resize(np.flipud(self.vis_spec),
                                  (round(self.vis_spec.shape[1] * self.image_stream.scale_factor),
                                   round(self.vis_spec.shape[0] * self.image_stream.scale_factor)))
        spec_excerpt = cm.viridis(spec_excerpt)[:, :, :3]
        spec_img = np.array((spec_excerpt * 255), dtype=np.uint8)

        return score_img, spec_img

    def stop_playing(self):
        self.is_piece_end = True

    def write_video(self, observations, audio):
        print('Storing Video...')
        observations = [cv2.cvtColor(o, cv2.COLOR_RGB2BGR) for o in observations]

        if self.audio_path is None:
            fname = "Live"
        else:
            fname = os.path.splitext(os.path.basename(self.audio_path))[0]

        if self.score_path is None:
            fname += "_camera"

        create_video(observations, audio, fname, FPS, SAMPLE_RATE, tag="", path="../videos")
        print('Done!')
