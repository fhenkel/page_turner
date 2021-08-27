import cv2
import torch
import numpy as np

from cyolo_score_following.models.yolo import load_pretrained_model
from cyolo_score_following.utils.data_utils import load_piece_for_testing, SAMPLE_RATE, FPS, FRAME_SIZE, HOP_SIZE
from cyolo_score_following.utils.general import xywh2xyxy
from cyolo_score_following.utils.video_utils import plot_box, plot_line
from cyolo_score_following.utils.general import load_wav
from collections import Counter
from scipy import interpolate
import matplotlib.cm as cm
import os


class Score_Audio_Prediction:
    def __init__(self, param_path, live_audio=None, audio_path=None, live_score=None, score_path=None,
                 gt_only=False, page=None):
        self.gt_only = gt_only
        self.live_audio = live_audio
        self.audio_path = audio_path
        self.live_score = live_score
        self.score_path = score_path
        self.org_scores, self.score, self.signal_np, self.systems, self.interpol_fnc, self.pad, self.scale_factor,\
            self.n_pages = [None] * 8
        self.load_essentials()
        self.page_plots = self.load_page_plots()


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network, criterion = load_pretrained_model(param_path)

        print(self.network)
        print("Putting model to %s ..." % device)
        self.network.to(device)
        print("Number of parameters:", sum(p.numel() for p in self.network.parameters() if p.requires_grad))
        self.network.eval()

        self.signal = torch.from_numpy(self.signal_np).to(device)
        self.score_tensor = torch.from_numpy(self.score).unsqueeze(1).to(device)

        self.from_ = 0
        self.to_ = FRAME_SIZE

        self.hidden = None
        self.frame_idx = 0

        self.actual_page = 0
        self.track_page = page
        self.start_ = None
        self.vis_spec = None
        self.is_piece_end = False

    def load_essentials(self):
        if self.audio_path is not None:
            self.signal_np = self.load_audio()

        if self.score_path is not None:
            self.org_scores, self.score, self.systems, self.interpol_fnc, self.pad, self.scale_factor, self.n_pages = \
                self.load_score()
        else:
            pass

    def load_score(self, scale_width=416):
        npzfile = np.load(self.score_path, allow_pickle=True)

        org_scores = npzfile["sheets"]
        coords, systems = list(npzfile["coords"]), list(npzfile['systems'])

        synthesized = npzfile['synthesized'].item()
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
        scale_factor = scores[0].shape[0] / scale_width

        for score in scores:
            scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

        score = np.stack(scaled_score)

        org_scores_rgb = []
        for org_score in org_scores:
            org_score = np.array(org_score, dtype=np.float32) / 255.

            org_scores_rgb.append(cv2.cvtColor(org_score, cv2.COLOR_GRAY2BGR))

        return org_scores_rgb, score, systems, interpol_fnc, pad1, scale_factor, n_pages

    def load_audio(self):
        signal = load_wav(self.audio_path, sr=SAMPLE_RATE)
        return signal

    def load_page_plots(self):
        page_plots = []
        for curr_page in range(self.n_pages):
            img_pred = cv2.cvtColor(self.org_scores[curr_page], cv2.COLOR_RGB2BGR)
            page_plots.append(np.array((img_pred * 255), dtype=np.uint8))

        return page_plots

    def end_of_piece(self):
        return self.to_ > self.signal_np.shape[-1]

    def get_next_images(self):
        true_position = np.array(self.interpol_fnc(self.frame_idx), dtype=np.float32)

        if self.actual_page != int(true_position[-1]):
            self.hidden = None

        self.actual_page = int(true_position[-1])
        system = self.systems[int(true_position[2])]
        true_position = true_position[:2]

        if self.track_page is None or self.actual_page == self.track_page:
            self.start_ = self.from_ if self.start_ is None else self.start_

            with torch.no_grad():
                sig_excerpt = self.signal[self.from_:self.to_]
                spec_frame = self.network.compute_spec([sig_excerpt], tempo_aug=False)[0]

                z, self.hidden = self.network.conditioning_network.get_conditioning(spec_frame, hidden=self.hidden)
                inference_out, pred = self.network.predict(self.score_tensor[self.actual_page:self.actual_page + 1], z)

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

            height = system['h'] / 2
            center_y, center_x = true_position

            # that is what i need for the app
            img_pred = cv2.cvtColor(self.org_scores[self.actual_page], cv2.COLOR_RGB2BGR)

            plot_line([center_x - self.pad, center_y, height], img_pred, label="GT",
                      color=(0.96, 0.63, 0.25), line_thickness=2)

            if not self.gt_only:
                img_pred = plot_box([x1, y1, x2, y2], img_pred, label="Pred", color=(0, 0, 1), line_thickness=2)

            score_img_data = np.array((img_pred * 255), dtype=np.uint8)
            spec_excerpt = cv2.resize(np.flipud(self.vis_spec), (round(self.vis_spec.shape[1] * self.scale_factor),
                                                            round(self.vis_spec.shape[0] * self.scale_factor)))
            spec_excerpt = cm.viridis(spec_excerpt)[:, :, :3]
            spec_excerpt = np.array((spec_excerpt * 255), dtype=np.uint8)


        else:
            if self.start_ is not None:
                # avoid moving back to the page
                # (in case repetitions span across multiple pages, shouldn't happen in msmd)
                self.is_piece_end = True
                return

        self.from_ += HOP_SIZE
        self.to_ += HOP_SIZE
        self.frame_idx += 1

        self.is_piece_end = self.end_of_piece()
        return score_img_data, spec_excerpt


