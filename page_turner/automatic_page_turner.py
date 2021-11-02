
import pyqtgraph as pg

from page_turner.dialog import DialogWindow
from page_turner.prediction import ScoreAudioPrediction
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):

        super().__init__()
        self.setWindowTitle("Automatic Page Turner")

        height = self.screen().size().height()
        self.resize(self.screen().size().width(), height)

        self.param_path, self.audio_path, self.score_path, self.n_pages = None, None, None, None

        vert_layout = QtWidgets.QVBoxLayout()

        # initializes the big score image
        self.Scores_graphics = pg.GraphicsView()
        self.Scores_graphics.setBackground(None)

        self.Scores_graphics.setMinimumHeight(600)
        self.Scores_graphics.show()
        self.score_image = None

        scores_view = pg.ViewBox(defaultPadding=0)
        scores_view.invertY()
        self.Scores_graphics.setCentralItem(scores_view)
        scores_view.setAspectLocked(True)

        self.score_img = pg.ImageItem(border="l", levels=(0, 255), axisOrder='row-major')
        scores_view.addItem(self.score_img)

        # initializes the audio image
        self.audio_graphics = pg.GraphicsView()
        self.audio_graphics.setBackground(None)
        self.audio_graphics.setMaximumHeight(150)
        self.audio_graphics.setMinimumHeight(150)
        self.audio_graphics.show()

        audio_view = pg.ViewBox(defaultPadding=0)
        audio_view.invertY()
        self.audio_graphics.setCentralItem(audio_view)
        audio_view.setAspectLocked(True)

        self.audio_img = pg.ImageItem(border="l", levels=(0, 255), axisOrder='row-major')
        audio_view.addItem(self.audio_img)

        vert_layout.addWidget(self.Scores_graphics)
        vert_layout.addWidget(self.audio_graphics)

        # input data dialog
        self.dialog = DialogWindow()

        # start/stop buttons
        self.start_tracking_button = QtWidgets.QPushButton("Start tracking...")
        self.start_tracking_button.clicked.connect(self.start_tracking)

        self.stop_tracking_button = QtWidgets.QPushButton("Stop tracking...")
        self.stop_tracking_button.clicked.connect(self.stop_tracking)
        vert_layout.addWidget(self.start_tracking_button)
        vert_layout.addWidget(self.stop_tracking_button)

        # image predictor will be an instance of ScoreAudioPrediction to process score and audio
        self.image_predictor = None
        self.timer = None

        container = QtWidgets.QWidget()
        container.setLayout(vert_layout)

        # Set the central widget of the Window.
        self.setCentralWidget(container)

    def stop_tracking(self):
        if self.timer is not None:
            self.timer.stop()

        if self.image_predictor is not None:
            self.image_predictor.stop_playing()

    def start_tracking(self):
        self.stop_tracking()

        self.dialog.exec()

        self.param_path = self.dialog.model_path  # path leading to the chosen model loaded from a .pt file
        self.audio_path = self.dialog.audio_path  # None (if live) or the path to the .wav file of the audio
        self.score_path = self.dialog.score_path  # None (if live) or the path to the .npz file of the score
        self.n_pages = self.dialog.n_pages

        self.image_predictor = ScoreAudioPrediction(self.param_path, audio_path=self.audio_path,
                                                    score_path=self.score_path, n_pages=self.n_pages)

        self.image_predictor.start()

        # start update timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start()

    def update_visualization(self):
        self.score_image, audio_image = self.image_predictor.get_next_images()

        self.score_img.setImage(self.score_image)
        self.audio_img.setImage(audio_image)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
