import cv2

import numpy as np

from page_turner.camera import Camera
from page_turner.config import *
from page_turner.utils import find_system_ys
from multiprocessing import Pipe


class ImageStream(object):

    def __init__(self, score_path=None, n_pages=None):
        self.score_path = score_path

        self.camera = None
        self.org_scores, self.score, self.system_ys = None, None, None
        if self.score_path is None:
            self.p_output, self.p_input = Pipe()
            self.camera = Camera(pipe=self.p_output)
            self.camera.start()

            self.pad = PADDING
            self.scale_factor = SCALE_FACTOR
            self.n_pages = n_pages if n_pages is not None else np.infty
            self.get_camera_input(wait_for_input=True)

        else:
            self.org_scores, self.score, self.system_ys, self.pad, self.scale_factor, self.n_pages = self.load_score()

    @property
    def camera_input(self):
        return self.camera is not None

    def more_pages(self, page):
        return page + 1 < self.n_pages

    def get_camera_input(self, wait_for_input=False):
        assert self.camera_input

        if wait_for_input or self.p_input.poll():
            org_score, scaled_score, system_ys = self.p_input.recv()
            self.org_scores = np.asarray([org_score])
            self.score = np.asarray([scaled_score])
            self.system_ys = np.asarray([system_ys])

    def load_score(self):
        """
        This function loads all variables required from the score.
        :return:
        """
        npzfile = np.load(self.score_path, allow_pickle=True)

        org_scores = npzfile["sheets"]

        n_pages, h, w = org_scores.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

        # Determine padding
        pad = ((0, 0), (0, 0), (pad1, pad2))

        # Add padding
        padded_scores = np.pad(org_scores, pad, mode="constant", constant_values=255)

        scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

        # scale scores
        scaled_score = []
        scale_factor = scores[0].shape[0] / SCALE_WIDTH

        for score in scores:
            scaled_score.append(cv2.resize(score, (SCALE_WIDTH, SCALE_WIDTH), interpolation=cv2.INTER_AREA))

        score = np.stack(scaled_score)

        org_scores_rgb = []
        system_ys = []
        for org_score in org_scores:
            org_score = cv2.cvtColor(np.array(org_score, dtype=np.float32) / 255., cv2.COLOR_GRAY2BGR)
            system_ys.append(find_system_ys(org_score))
            org_scores_rgb.append(org_score)

        return org_scores_rgb, score, system_ys, pad1, scale_factor, n_pages

    def get(self, page):

        if self.camera_input:
            page = 0
            self.get_camera_input()

        return self.org_scores[page], self.score[page], self.system_ys[page]

    def close(self):

        if self.camera_input:
            self.camera.terminate()
            self.camera.join()


