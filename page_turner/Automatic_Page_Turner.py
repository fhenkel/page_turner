# -*- coding: utf-8 -*-
import os.path
import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMenu, QAction
from PyQt5.QtCore import QTimer
from pyqtgraph import PlotWidget
from pyqtgraph.Qt import QtCore
import pyqtgraph.ptime as ptime
import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import (
    QFileDialog,
    QWidget,
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QStackedLayout,
)

from prediction import Score_Audio_Prediction

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Automatic Page Turner")
        self.resize(1000, 900)

        vert_layout = QVBoxLayout()

        # with open('test_array.npy', 'rb') as f:
        #     data = np.load(f)
        # data = np.array((data*255), dtype=np.uint8)

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

        # data_score = np.ones((1181, 835, 3)) * 128
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
        self.updateTime = ptime.time()
        self.fps = 0

        container = QWidget()
        container.setLayout(vert_layout)

        # Set the central widget of the Window.
        self.setCentralWidget(container)

        self._create_menu_bar()
        self.path = None
        self.choose_piece.triggered.connect(self.choose_piece_dir)

    def choose_piece_dir(self):
        print(os.getcwd())
        default_path = "/home/stephanie/Documents/Studium/Automatic_Page_Turning/cyolo_score_following/data/msmd"
        curr_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select one piece', default_path,
                                                             "Wav files (*.wav)",
                                                             options=QFileDialog.DontUseNativeDialog)
        print(self.path)
        param_path = os.path.join('..', 'models', 'test_model', 'best_model.pt')
        test_dir, test_piece = os.path.split(curr_path)
        test_piece = os.path.splitext(test_piece)[0]
        self.image_predictor = Score_Audio_Prediction(param_path, test_dir, test_piece, scale_width=416, gt_only=True, page=None)
        self.updateData()

    def updateData(self):
        score_image, audio_image = self.image_predictor.get_next_images()
        self.score_img.setImage(score_image)
        self.audio_img.setImage(audio_image)
        # creating a qtimer
        QTimer.singleShot(1, self.updateData)
        # getting current time
        now = ptime.time()
        # temporary fps
        fps2 = 1.0 / (now - self.updateTime)
        # updating the time
        self.updateTime = now
        # setting original fps value
        self.fps = self.fps * 0.9 + fps2 * 0.1

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        self.menu_piece = QMenu("&Piece", self)
        menu_bar.addMenu(self.menu_piece)
        self.choose_piece = QAction("&Choose Piece", self)
        self.menu_piece.addAction(self.choose_piece)

        self.menu_view = QMenu("&View", self)
        menu_bar.addMenu(self.menu_view)

        self.menu_show_ground_truth = self.menu_view.addMenu("view ground truth")
        self.action_ground_truth_Note_Level = QAction("&Note Level", self)
        self.action_ground_truth_Note_Level.triggered.connect(
            lambda: self.activate_action(self.action_ground_truth_Note_Level))

        self.action_ground_truth_Bar_Level = QAction("&Bar Level", self)
        self.action_ground_truth_Bar_Level.triggered.connect(
            lambda: self.activate_action(self.action_ground_truth_Bar_Level))
        self.action_ground_truth_System_Level = QAction("&System Level", self)
        self.action_ground_truth_System_Level.triggered.connect(
            lambda: self.activate_action(self.action_ground_truth_System_Level))
        self.menu_show_ground_truth.addAction(self.action_ground_truth_Note_Level)
        self.menu_show_ground_truth.addAction(self.action_ground_truth_Bar_Level)
        self.menu_show_ground_truth.addAction(self.action_ground_truth_System_Level)

        self.menu_prediction_level = self.menu_view.addMenu("prediction level")
        self.action_prediction_Note_level = QAction("&Note Level", self)
        self.action_prediction_Note_level.triggered.connect(
            lambda: self.activate_action(self.action_prediction_Note_level))
        self.action_prediction_Bar_level = QAction("&Bar Level", self)
        self.action_prediction_Bar_level.triggered.connect(
            lambda: self.activate_action(self.action_prediction_Bar_level))
        self.action_prediction_System_level = QAction("&System Level", self)
        self.action_prediction_System_level.triggered.connect(
            lambda: self.activate_action(self.action_prediction_System_level))
        self.menu_prediction_level.addAction(self.action_prediction_Note_level)
        self.menu_prediction_level.addAction(self.action_prediction_Bar_level)
        self.menu_prediction_level.addAction(self.action_prediction_System_level)

        self.menu_model = QMenu("&Model", self)
        menu_bar.addMenu(self.menu_model)
        self.actionDefault = QAction("&Default", self)
        self.actionDefault.triggered.connect(lambda: self.activate_action(self.actionDefault))
        self.menu_model.addAction(self.actionDefault)

    def activate_action(self, action):
        myFont = QtGui.QFont()
        myFont.setBold(not action.font().bold())
        action.setFont(myFont)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
