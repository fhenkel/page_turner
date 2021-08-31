from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import math
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QPushButton,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QScrollArea,
    QFrame
)

from page_turner.dialog import DialogWindow
from page_turner.prediction import ScoreAudioPrediction


class MainWindow(QMainWindow):
    def __init__(self):

        super().__init__()
        self.setWindowTitle("Automatic Page Turner")
        self.resize(1500, 900)

        self.param_path, self.audio_path, self.score_path = None, None, None
        self.ground_truth_box_states, self.prediction_box_states = None, None

        self.last_page = 0
        # used for page-click event
        self.other_page_shown = False

        horiz_layout = QHBoxLayout()
        vert_layout = QVBoxLayout()

        # This scrollable area will show all score pages in small
        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setGeometry(10, 1000, 400, 50)
        self.scroll_widget = QWidget()
        
        self.page_grid = QGridLayout()
        self.page_graphics = []
        self.image_items = []

        # initializes the big score image
        self.Scores_graphics = pg.GraphicsView()
        self.Scores_graphics.setBackground(None)
        self.Scores_graphics.setGeometry(10, 20, 650, 50)
        self.Scores_graphics.setMinimumSize(600, 50)
        self.Scores_graphics.setMaximumWidth(50)
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
        self.audio_graphics.setGeometry(10, 690, 630, 101)
        self.audio_graphics.setMinimumSize(630, 100)
        self.audio_graphics.setMaximumSize(650, 150)
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
        self.start_tracking_button = QPushButton("Start tracking...")
        self.start_tracking_button.clicked.connect(self.start_tracking)

        self.stop_tracking_button = QPushButton("Stop tracking...")
        self.stop_tracking_button.clicked.connect(self.stop_tracking)
        vert_layout.addWidget(self.start_tracking_button)
        vert_layout.addWidget(self.stop_tracking_button)

        # image predictor will be an instance of ScoreAudioPrediction
        # to process score and audio
        self.image_predictor = None
        self.timer = None

        horiz_layout.addLayout(vert_layout)
        # if self.audio_path is not None:
        horiz_layout.addWidget(self.scroll)
        container = QWidget()
        container.setLayout(horiz_layout)

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

        # list containing the boolean values for showing different levels of ground truth [note, bar, system]
        self.ground_truth_box_states = self.dialog.ground_truth_box_states

        # list containing the boolean values for showing different levels of prediction [note, bar, system]
        self.prediction_box_states = self.dialog.prediction_box_states
        self.create_prediction_object()

    def load_pages(self):
        """
        This function creates for each score page an image that is shown
        in the self.page_grid
        :return:
        """

        for i, page in enumerate(self.image_predictor.page_plots):
            gv = pg.GraphicsView()
            gv.setFixedSize(250, 400)
            gv.show()

            scores_view = pg.ViewBox(defaultPadding=0.001)
            scores_view.invertY()
            gv.setCentralItem(scores_view)
            scores_view.setAspectLocked(True)

            self.image_items.append(pg.ImageItem(image=page, border={"color": "l", "width": 2}, levels=(0, 255),
                                                 axisOrder='row-major'))

            # first page will get a red frame and be bigger, since it will be the first tracked page
            if i == 0:
                gv.setFixedSize(300, 450)
                self.image_items[i].setBorder({"color": "r", "width": 2})
            scores_view.addItem(self.image_items[i])

            self.page_graphics.append(gv)

            # decrease click radius from 2 to 1, otherwise it will always choose the same one
            self.page_graphics[i].scene().setClickRadius(1)
            self.page_graphics[i].scene().sigMouseClicked.connect(lambda obj, index=i: self.show_clicked_page(index))
            self.page_grid.addWidget(self.page_graphics[i], math.floor(i/2), i % 2)

        self.scroll_widget.setLayout(self.page_grid)
        self.scroll.setWidget(self.scroll_widget)

        # self.scroll.update()
        # self.scroll_widget.update()

    def show_clicked_page(self, index):
        """
        This function changes the score page which is shown in big, to
        the small one in the self.page_grid clicked on.
        :param index: The index of the clicked image
        :return:
        """
        if index == self.image_predictor.actual_page:
            self.other_page_shown = False
            self.score_img.setImage(self.score_image)
        else:
            self.other_page_shown = True
            self.score_img.setImage(self.image_predictor.page_plots[index])

    def create_prediction_object(self):
        """
        This function creates a new instance of ScoreAudioPrediction, restarts the timer
        and starts the image prediction process and its visualization updates.
        :return:
        """

        # reset page elements
        self.page_graphics = []
        self.image_items = []

        for i in reversed(range(self.page_grid.count())):
            self.page_grid.itemAt(i).widget().setParent(None)

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
        """
        This function updates the visualization of the audio and of the prediction in the score:
        It increases the size of the currently tracked page and adds a red frame.
        If another page is chosen to be shown in big due to an click event, then it
        will not change the big image to the current tracked page.
        :return:
        """
        self.score_image, audio_image = self.image_predictor.get_next_images()
        if self.last_page != self.image_predictor.actual_page:
            self.page_graphics[self.last_page].setFixedSize(250, 400)
            self.image_items[self.last_page].setBorder({"color": "l", "width": 2})
            self.last_page = self.image_predictor.actual_page
            self.image_items[self.last_page].setBorder({"color": "r", "width": 2})
            self.page_graphics[self.last_page].setFixedSize(300, 450)

        if not self.other_page_shown:
            self.score_img.setImage(self.score_image)

        self.audio_img.setImage(audio_image)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
