import cv2
import multiprocessing
import subprocess

import numpy as np

from page_turner.config import *
from page_turner.utils import find_system_ys


class Camera(multiprocessing.Process):

    def __init__(self, pipe):
        multiprocessing.Process.__init__(self)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCORE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCORE_WIDTH)
        self.pipe = pipe

        # set custom focus for camera (probably camera specific, please change accordingly)
        subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=focus_auto=0"])
        subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=focus_absolute=30"])

    def run(self):

        while True:
            self.take_picture()

    def take_picture(self):
        ret, frame = self.cap.read()
        org_frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2GRAY)

        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)

        # crop score
        frame = frame[70:-100, 0:-45]
        org_frame = org_frame[70:-100, 0:-45]

        # resize
        frame = cv2.resize(frame, (SCORE_WIDTH, SCORE_HEIGHT), interpolation=cv2.INTER_AREA)
        org_frame = cv2.resize(org_frame, (SCORE_WIDTH, SCORE_HEIGHT), interpolation=cv2.INTER_AREA)

        # Add padding
        pad = ((0, 0), (PADDING, PADDING))
        padded_score = np.pad(frame, pad, mode="constant", constant_values=255)

        score = 1 - np.array(padded_score, dtype=np.float32) / 255.

        scaled_score = cv2.resize(score, (SCALE_WIDTH, SCALE_WIDTH), interpolation=cv2.INTER_AREA)

        org_score = np.array(org_frame, dtype=np.float32) / 255.

        system_ys = find_system_ys(org_score, thicken_lines=True)

        self.pipe.send((org_score, scaled_score, system_ys))

    def terminate(self):
        self.cap.release()
        super(Camera, self).terminate()


if __name__ == "__main__":
    from multiprocessing import Pipe

    p_output, p_input = Pipe()
    camera = Camera(pipe=p_output)

    camera.start()

    while True:
        if p_input.poll():
            org_img, scaled_img, system_ys = p_input.recv()

            for s in system_ys:
                cv2.line(org_img, (0, s[0]), (org_img.shape[1], s[0]), color=(0, 255, 0))
                cv2.line(org_img, (0, s[1]), (org_img.shape[1], s[1]), color=(0, 255, 0))

            cv2.imshow('Original Image', org_img)
            cv2.imshow('Scaled Image', scaled_img)

        cv2.waitKey(1)

