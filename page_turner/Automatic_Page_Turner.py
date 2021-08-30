
import os.path
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMenu, QAction
from pyqtgraph import PlotWidget
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import numpy as np
import math
from PyQt5.QtWidgets import (
    QFileDialog,
    QWidget,
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
)

from prediction import ScoreAudioPrediction


class MainWindow(QMainWindow):
    def __init__(self, audio_path, score_path, model_path, ground_truth_box_states,
                            prediction_box_states):
        super().__init__()
        self.default_path = os.path.join('..', 'demo_piece')
        self.param_path = model_path
        self.audio_path = audio_path
        self.score_path = score_path
        self.ground_truth_box_states = ground_truth_box_states
        self.prediction_box_states = prediction_box_states
        print(audio_path, score_path, model_path, ground_truth_box_states,
                            prediction_box_states)
        self.last_page = 0

        self.setWindowTitle("Automatic Page Turner")
        self.resize(1500, 900)

        horiz_layout = QHBoxLayout()
        vert_layout = QVBoxLayout()

        self.page_grid = QGridLayout()
        self.page_graphics = []
        self.image_items = []

        self.Scores_graphics = pg.GraphicsView()
        self.Scores_graphics.setBackground(None)
        self.Scores_graphics.setGeometry(10, 20, 650, 50)
        self.Scores_graphics.setMinimumSize(600, 50)
        self.Scores_graphics.setMaximumWidth(50)
        self.Scores_graphics.show()

        scores_view = pg.ViewBox(defaultPadding=0)
        scores_view.invertY()
        self.Scores_graphics.setCentralItem(scores_view)
        scores_view.setAspectLocked(True)

        self.score_img = pg.ImageItem(border="l", levels=(0, 255), axisOrder='row-major')
        scores_view.addItem(self.score_img)

        self.audio_graphics = pg.GraphicsView()
        self.audio_graphics.setBackground(None)
        self.audio_graphics.setGeometry(10, 690, 630, 101)
        self.audio_graphics.setMinimumSize(630, 100)
        self.audio_graphics.setMaximumSize(650, 150)
        self.audio_graphics.show()

        audio_view = pg.ViewBox(defaultPadding=0)
        audio_view.invertY()
        self.audio_graphics.setCentralItem(audio_view)
        audio_view.setAspectLocked(True)

        # data_audio = np.ones((390, 200)) * 128
        self.audio_img = pg.ImageItem(border="l", levels=(0, 255), axisOrder='row-major')
        audio_view.addItem(self.audio_img)

        vert_layout.addWidget(self.Scores_graphics)
        vert_layout.addWidget(self.audio_graphics)
        self.image_predictor = None
        self.timer = None

        horiz_layout.addLayout(vert_layout)
        horiz_layout.addLayout(self.page_grid)
        container = QWidget()
        container.setLayout(horiz_layout)

        # Set the central widget of the Window.
        self.setCentralWidget(container)

        self.create_prediction_object()

    def load_pages(self):
        for i, page in enumerate(self.image_predictor.page_plots):
            self.page_graphics.append(pg.GraphicsView())
            self.page_graphics[i].setBackground(None)
            self.page_graphics[i].setMinimumSize(250, 50)
            self.page_graphics[i].setMaximumWidth(50)
            self.page_graphics[i].show()

            scores_view = pg.ViewBox(defaultPadding=0.001)
            scores_view.invertY()
            self.page_graphics[i].setCentralItem(scores_view)
            scores_view.setAspectLocked(True)

            self.image_items.append(pg.ImageItem(image=page, border={"color": "l", "width": 2}, levels=(0, 255), axisOrder='row-major'))
            if i == 0:
                self.page_graphics[i].setMinimumSize(300, 50)
                self.image_items[i].setBorder({"color": "r", "width": 2})
            scores_view.addItem(self.image_items[i])

            self.page_grid.addWidget(self.page_graphics[i], math.floor(i/2), i % 2)

    def create_prediction_object(self):
        # TODO fix, now only called when audio is selected

        # stop old tracking
        if self.timer is not None:
            self.timer.stop()

        if self.image_predictor is not None:
            self.image_predictor.stop_playing()
            self.image_predictor = None

        # reset page elements
        self.page_graphics = []
        self.image_items = []

        print(self.score_path)
        print(self.audio_path)
        self.image_predictor = ScoreAudioPrediction(self.param_path, audio_path=self.audio_path,
                                                    score_path=self.score_path, gt_only=False, page=None)

        self.image_predictor.start()
        self.load_pages()

        # start update timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start()

    def update_visualization(self):
        score_image, audio_image = self.image_predictor.get_next_images()
        if self.last_page != self.image_predictor.actual_page:
            self.page_graphics[self.last_page].setMinimumSize(250, 50)
            self.image_items[self.last_page].setBorder({"color": "l", "width": 2})
            self.last_page = self.image_predictor.actual_page
            self.image_items[self.last_page].setBorder({"color": "r", "width": 2})
            self.page_graphics[self.last_page].setMinimumSize(300, 50)

        self.score_img.setImage(score_image)
        self.audio_img.setImage(audio_image)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
