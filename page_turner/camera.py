import cv2
import multiprocessing
import numpy as np


class Camera(multiprocessing.Process):

    def __init__(self, pipe, scale_width=416):
        multiprocessing.Process.__init__(self)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1181)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 835)
        self.scale_width = scale_width
        self.pipe = pipe

        self.continue_running = True

    def run(self):

        while self.continue_running:
            self.take_picture()

    def take_picture(self):
        ret, frame = self.cap.read()
        org_frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2GRAY)

        # img_edges = cv2.Canny(frame, 100, 100, apertureSize=3)
        # lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
        #
        # angles = []
        #
        # for [[x1, y1, x2, y2]] in lines:
        #     angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        #     angles.append(angle)
        #
        # median_angle = np.median(angles)
        #
        # frame = ndimage.rotate(frame, median_angle)
        # org_frame = ndimage.rotate(org_frame, median_angle)

        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # frame = cv2.fastNlMeansDenoising(frame, 11, 31, 9)

        # frame = frame[100:-280, 50:-50]
        # frame = frame[120:-180, 20:-75]
        frame = frame[70:-100, 0:-45]
        org_frame = org_frame[70:-100, 0:-45]
        frame = cv2.resize(frame, (835, 1181), interpolation=cv2.INTER_AREA)
        org_frame = cv2.resize(org_frame, (835, 1181), interpolation=cv2.INTER_AREA)

        h, w = frame.shape

        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

        # Determine padding
        pad = ((0, 0), (pad1, pad2))

        # Add padding
        padded_score = np.pad(frame, pad, mode="constant", constant_values=255)

        score = 1 - np.array(padded_score, dtype=np.float32) / 255.

        scaled_score = cv2.resize(score, (self.scale_width, self.scale_width), interpolation=cv2.INTER_AREA)

        # self.org_score = cv2.cvtColor(np.array(org_frame, dtype=np.float32) / 255., cv2.COLOR_GRAY2BGR)
        org_score = np.array(org_frame, dtype=np.float32) / 255.

        self.pipe.send((org_score, scaled_score))

    def stop_running(self):
        self.continue_running = False


if __name__ == "__main__":
    from multiprocessing import Pipe

    p_output, p_input = Pipe()
    camera = Camera(pipe=p_output, scale_width=416)

    camera.start()

    while True:
        if p_input.poll():
            org_img, scaled_img = p_input.recv()

            cv2.imshow('Original Image', org_img)
            cv2.imshow('Scaled Image', scaled_img)

        cv2.waitKey(1)

