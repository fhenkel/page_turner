import copy
import cv2
import serial
import torch
import threading

import matplotlib.cm as cm
import numpy as np

from cyolo_score_following.models.yolo import load_pretrained_model
from cyolo_score_following.utils.data_utils import SAMPLE_RATE, FPS, FRAME_SIZE, HOP_SIZE
from cyolo_score_following.utils.general import xywh2xyxy
from cyolo_score_following.utils.video_utils import plot_box, create_video

from page_turner.audio_stream import AudioStream
from page_turner.config import *
from page_turner.image_stream import ImageStream


class ScoreAudioPrediction(threading.Thread):

    def __init__(self, param_path,  audio_path=None, score_path=None, n_pages=None):
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

    def get_next_images(self):
        return self.score_img, self.spec_img

    def get_best_prediction(self, predictions, systems, th=0.5, start_from_top=False):
        _, idx = torch.sort(predictions[:, 4], descending=True)
        sorted_predictions = predictions[idx]

        sorted_predictions[:, :4] *= self.image_stream.scale_factor
        sorted_predictions[:, 0] -= self.image_stream.pad



        # x1, y1, x2, y2 = xywh2xyxy(sorted_predictions[:, :4]).cpu().numpy().T
        # return [x1[0], y1[0], x2[0], y2[0]]

        # filtered_predictions = sorted_predictions[sorted_predictions[:, 4] > th]
        # filtered_predictions = sorted_predictions[sorted_predictions[:, 4] > th]
        filtered_predictions = sorted_predictions
        # filtered_predictions = sorted_predictions
        # filtered_predictions = sorted_predictions[:]

        best = self.previous_prediction
        try:
            if filtered_predictions.shape[0] > 0:

                x1, y1, x2, y2 = xywh2xyxy(filtered_predictions[:, :4]).cpu().numpy().T

                y_means = y1 + (y2 - y1) / 2
                in_first_system = (y_means >= systems[0][0]) & (y_means <= systems[0][1])
                start_in_front = (x1 < SCORE_WIDTH*0.3)
                # if self.previous_prediction is not None:
                if not start_from_top and self.previous_prediction is not None:
                    previous_y_mean = self.previous_prediction[1] \
                                      + (self.previous_prediction[3] - self.previous_prediction[1])/2

                    curr_system_idx = np.argwhere((systems[:, 0] <= previous_y_mean) & (systems[:, 1] >= previous_y_mean)).item()
                    curr_system = systems[curr_system_idx]

                    next_system = systems[min(curr_system_idx + 1, len(systems) - 1)]
                    # only move to the right
                    # move_right = (x1 >= self.previous_prediction[0] - 20)
                    move_right = True #(x1 >= self.previous_prediction[0] - 100) #& (x1 <= self.previous_prediction[0] + 100)
                    # stay_within_system = (self.previous_prediction[1] - 20 <= y1) & (self.previous_prediction[1] + 20 >= y1)
                    stay_within_system = (y_means >= curr_system[0]) & (y_means <= curr_system[1])
                    # move_to_next_system = (self.previous_prediction[1] + 20 <= y1)

                    # only allow jump if prediction is past certain x-location
                    allow_move_to_next = SCORE_WIDTH*0.7 < self.previous_prediction[0] and (next_system != curr_system_idx).all()
                    move_to_next_system = allow_move_to_next & (y_means >= next_system[0]) & (y_means <= next_system[1])

                    indices = (move_right & stay_within_system) | (move_to_next_system & start_in_front)

                    if any(indices):
                        x1 = x1[indices]
                        x2 = x2[indices]
                        y1 = y1[indices]
                        y2 = y2[indices]
                        best = [x1[0], y1[0], x2[0], y2[0]]

                else:

                    indices = (in_first_system & start_in_front) & start_from_top
                    if any(indices):
                        x1 = x1[indices]
                        x2 = x2[indices]
                        y1 = y1[indices]
                        y2 = y2[indices]
                        # best = [x1[0], y1[0], x2[0], y2[0]]

                        best = [x1[0], y1[0], x2[0], y2[0]]

        except:
            x1, y1, x2, y2 = xywh2xyxy(sorted_predictions[:, :4]).cpu().numpy().T
            best = [x1[0], y1[0], x2[0], y2[0]]


        # if filtered_predictions.shape[0] == 0 or best is None:
        #     # else:
        #
        #     th -= 0.1
        #
        #     if th <= 0:
        #         self.previous_prediction = None
        #     # if self.previous_prediction is None:
        #     #     th -= 0.1
        #     # self.previous_prediction = None
        #
        #     best = self.get_best_prediction(predictions, systems, th, start_from_top=start_from_top)

        # else:
        #     self.cooldown += 1
        #
        #     if self.cooldown > 20:
        #         x1, y1, x2, y2 = xywh2xyxy(sorted_predictions[:, :4]).cpu().numpy().T
        #         best = [x1[0], y1[0], x2[0], y2[0]]
        #         self.cooldown = 0

        return best

    def run(self):
        self.is_piece_end = False
        signal = None

        from_ = 0
        to_ = FRAME_SIZE

        org_score, score, system_ys = self.image_stream.get(self.actual_page)

        hidden = None
        frame_idx = 0

        curr_y = 0
        curr_x = 0
        observations = []

        self.previous_prediction = None
        self.cooldown = 0
        page_turner_cooldown = 0
        start_from_top = True
        while not self.is_piece_end:

            frame = self.audio_stream.get()

            if frame is None:
                break

            signal = np.concatenate((signal, frame)) if signal is not None else frame

            if len(signal[from_:to_]) != FRAME_SIZE:
                continue

            if len(system_ys) > 0 and system_ys[-1][0] <= curr_y <= system_ys[-1][1] and curr_x > SCORE_WIDTH / 2:

                if self.image_stream.more_pages(self.actual_page) and page_turner_cooldown <= 0:
                    print('Turn page')
                    hidden = None
                    self.actual_page += 1
                    self.previous_prediction = None
                    page_turner_cooldown = 10
                    start_from_top = True

                    if self.image_stream.camera_input:
                        try:
                            with serial.Serial('/dev/ttyUSB0', 9600, timeout=1) as ser:
                                ser.write(serial.to_bytes([0xA0, 0x01, 0x01, 0xA2]))
                                ser.write(serial.to_bytes([0xA0, 0x01, 0x00, 0xA1]))
                        except:
                            print("Physical page turning not possible. Did you run sudo chown <user> /dev/ttyUSB0 ?")

            org_score, score, system_ys = self.image_stream.get(self.actual_page)

            # add channel and batch dimension
            score_tensor = torch.from_numpy(score).unsqueeze(0).unsqueeze(0).to(self.device)

            if page_turner_cooldown > 0:
                page_turner_cooldown -= 1
                print('cool down')
                hidden = None
                self.previous_prediction = None
                start_from_top = True

            with torch.no_grad():
                sig_excerpt = torch.from_numpy(signal[from_:to_]).float().to(self.device)
                spec_frame = self.network.compute_spec([sig_excerpt], tempo_aug=False)[0]

                z, hidden = self.network.conditioning_network.get_conditioning(spec_frame, hidden=hidden)
                inference_out, pred = self.network.predict(score_tensor, z)

            filtered_inference_out = inference_out[0, inference_out[0, :, -1] == 0].unsqueeze(0)

            best_prediction = self.get_best_prediction(filtered_inference_out[0], system_ys, start_from_top=start_from_top)

            if best_prediction is not None:
                x1, y1, x2, y2 = best_prediction
                curr_y = y1 + (y2 - y1) / 2
                curr_x = x1 + (x2 - x1) / 2

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
            self.vis_spec = np.zeros((spec_frame.shape[-1], 400))

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
