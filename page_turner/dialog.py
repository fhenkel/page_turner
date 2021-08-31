import os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QFileDialog, QDialogButtonBox


class DialogWindow(QDialog):
    def __init__(self):
        """
        This function initializes the dialog window for the page turner application
        """
        super().__init__()

        self.setWindowTitle("Please choose your setting")
        self.resize(652, 482)

        self.gridLayout = QtWidgets.QGridLayout()

        # setup ok button
        ok_button = QtWidgets.QPushButton("ok")
        ok_button.clicked.connect(self.open_main_window)

        self.gridLayout.addWidget(ok_button, 9, 4, 1, 1)

        # 'Piece' text part
        self.piece_label = QtWidgets.QLabel("Piece", self)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.piece_label.setFont(font)
        self.gridLayout.addWidget(self.piece_label, 0, 0, 1, 1)

        # 'Audio' text part and dropdown for audio
        self.audio_layout = QtWidgets.QVBoxLayout()
        self.audio_label = QtWidgets.QLabel("Audio", self)
        font = QtGui.QFont()
        font.setBold(True)
        self.audio_label.setFont(font)
        self.audio_label.setAlignment(QtCore.Qt.AlignCenter)
        self.audio_layout.addWidget(self.audio_label)
        self.audio_dropdown = QtWidgets.QComboBox()
        self.audio_dropdown.addItem("live")
        self.audio_dropdown.addItem("choose from library")
        self.audio_layout.addWidget(self.audio_dropdown)
        self.gridLayout.addLayout(self.audio_layout, 0, 2, 2, 1)

        # 'Score' text part and dropdown for score
        self.score_layout = QtWidgets.QVBoxLayout()
        self.score_label = QtWidgets.QLabel("Score", self)
        font = QtGui.QFont()
        font.setBold(True)
        self.score_label.setFont(font)
        self.score_label.setAlignment(QtCore.Qt.AlignCenter)
        self.score_layout.addWidget(self.score_label)
        self.score_dropdown = QtWidgets.QComboBox()
        self.score_dropdown.addItem("live")
        self.score_dropdown.addItem("choose from library")
        self.score_layout.addWidget(self.score_dropdown)
        self.gridLayout.addLayout(self.score_layout, 0, 4, 2, 1)

        # 'View' text part
        self.view_label = QtWidgets.QLabel("View", self)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.view_label.setFont(font)
        self.view_label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.gridLayout.addWidget(self.view_label, 3, 0, 1, 1)

        # 'Prediction' text part and the corresponding checkboxes
        self.view_prediction_layout = QtWidgets.QVBoxLayout()
        self.prediction_label = QtWidgets.QLabel("Prediction", self)
        font = QtGui.QFont()
        font.setBold(True)
        self.prediction_label.setFont(font)
        self.view_prediction_layout.addWidget(self.prediction_label)
        self.prediction_note_box = QtWidgets.QCheckBox("&note level")
        self.view_prediction_layout.addWidget(self.prediction_note_box)
        self.prediction_bar_box = QtWidgets.QCheckBox("&bar level")
        self.view_prediction_layout.addWidget(self.prediction_bar_box)
        self.prediction_system_box = QtWidgets.QCheckBox("&system level")
        self.view_prediction_layout.addWidget(self.prediction_system_box)
        self.gridLayout.addLayout(self.view_prediction_layout, 3, 4, 2, 1)

        # 'Ground Truth' text part and the corresponding checkboxes
        self.ground_truth_layout = QtWidgets.QVBoxLayout()
        self.ground_truth_label = QtWidgets.QLabel("Ground Truth", self)
        font = QtGui.QFont()
        font.setBold(True)
        self.ground_truth_label.setFont(font)
        self.ground_truth_layout.addWidget(self.ground_truth_label)
        self.gtruth_note_box = QtWidgets.QCheckBox("&note level")
        self.ground_truth_layout.addWidget(self.gtruth_note_box)
        self.gtruth_bar_box = QtWidgets.QCheckBox("&bar level")
        self.ground_truth_layout.addWidget(self.gtruth_bar_box)
        self.gtruth_system_box = QtWidgets.QCheckBox("&system level")
        self.ground_truth_layout.addWidget(self.gtruth_system_box)
        self.gridLayout.addLayout(self.ground_truth_layout, 3, 2, 2, 1)

        # 'Model' text part
        self.model_label = QtWidgets.QLabel("Model", self)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.model_label.setFont(font)
        self.model_label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.gridLayout.addWidget(self.model_label, 6, 0, 1, 1)

        # dropdown for model
        self.model_dropdown = QtWidgets.QComboBox()
        self.model_dropdown.addItem("default")
        self.model_dropdown.addItem("choose local")
        self.gridLayout.addWidget(self.model_dropdown, 6, 2, 1, 1)

        self.setLayout(self.gridLayout)

        ##################################################################
        # this contains all spacers to keep the correct layout
        piece_view_spacer = QtWidgets.QSpacerItem(20, 60, QtWidgets.QSizePolicy.Minimum,
                                                  QtWidgets.QSizePolicy.MinimumExpanding)
        self.gridLayout.addItem(piece_view_spacer, 1, 0, 2, 1)
        view_model_spacer = QtWidgets.QSpacerItem(20, 60, QtWidgets.QSizePolicy.Minimum,
                                                  QtWidgets.QSizePolicy.MinimumExpanding)
        self.gridLayout.addItem(view_model_spacer, 4, 0, 2, 1)

        column1_spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.MinimumExpanding,
                                               QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(column1_spacer, 1, 1, 1, 1)

        audio_gt_spacer = QtWidgets.QSpacerItem(20, 60, QtWidgets.QSizePolicy.Minimum,
                                                QtWidgets.QSizePolicy.MinimumExpanding)
        self.gridLayout.addItem(audio_gt_spacer, 2, 2, 1, 1)

        gt_model_dropd_spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum,
                                                      QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(gt_model_dropd_spacer, 5, 2, 1, 1)

        column2_spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.MinimumExpanding,
                                               QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(column2_spacer, 0, 3, 1, 1)

        score_pred_spacer = QtWidgets.QSpacerItem(20, 60, QtWidgets.QSizePolicy.Minimum,
                                                  QtWidgets.QSizePolicy.MinimumExpanding)
        self.gridLayout.addItem(score_pred_spacer, 2, 4, 1, 1)

        input_button_spacer = QtWidgets.QSpacerItem(20, 60, QtWidgets.QSizePolicy.Minimum,
                                                    QtWidgets.QSizePolicy.MinimumExpanding)
        self.gridLayout.addItem(input_button_spacer, 7, 0, 2, 1)
        ##################################################################
        # this initializes all important variables and connects the dropdowns
        self.audio_path = None
        self.score_path = None
        self.prediction_box_states = None
        self.ground_truth_box_states = None

        self.model_path = os.path.join('..', 'models', 'test_model', 'best_model.pt')
        self.window = None
        self.audio_dropdown.activated[str].connect(self.react_to_audio_dropdown)
        self.score_dropdown.activated[str].connect(self.react_to_score_dropdown)
        self.model_dropdown.activated[str].connect(self.react_to_model_dropdown)

    def open_main_window(self):
        """
        This function opens the main window of the page turner application
        with all parameters, when the button 'Start tracking' is clicked.
        :return:
        """
        self.prediction_box_states = [self.prediction_note_box.isChecked(), self.prediction_bar_box.isChecked(),
                                      self.prediction_system_box.isChecked()]
        self.ground_truth_box_states = [self.gtruth_note_box.isChecked(), self.gtruth_bar_box.isChecked(),
                                        self.gtruth_system_box.isChecked()]
        self.close()

    def react_to_audio_dropdown(self, text):
        """
        This function opens a dialog window to choose
        a wav file for the audio, when "choose from library" is clicked
        in the dropdown
        :param text: "live" or "choose from library"
        :return:
        """
        if text == "choose from library":
            self.choose_piece_dir("wav")
        elif text == "live":
            # set live input
            self.set_path(None, "wav")
        else:
            raise NotImplementedError

    def react_to_score_dropdown(self, text):
        """
            This function opens a dialog window to choose
            a npz file for the score, when "choose from library" is clicked
            in the dropdown
            :param text: "live" or "choose from library"
            :return:
        """

        if text == "choose from library":
            self.choose_piece_dir("npz")
        elif text == "live":
            # set live input
            self.set_path(None, "npz")
        else:
            raise NotImplementedError

    def react_to_model_dropdown(self, text):
        """
            This function opens a dialog window to choose
            a .pt file for the model, when "choose local" is clicked
            in the dropdown
            :param text: "default" or "choose local"
            :return: 
        """
        if text == "choose local":
            self.choose_piece_dir("pt", os.path.join('..', 'models'))
        elif text == "default":
            self.model_path = os.path.join('..', 'models', 'test_model', 'best_model.pt')
        else:
            raise NotImplementedError

    def choose_piece_dir(self, extension, path=None, ):
        """
        This function opens a dialog window to choose a file
        with a specific extension in the default path or in another chosen path.
        The full path of the chosen file is saved to the respective variable.
        :param extension: str of the extension the file must have options: ["wav", "npz", "pt"]
        :param path: path to open first in the dialog
        :return:
        """
        if path is not None:
            search_path = path
        else:
            search_path = os.path.join('..', 'demo_piece')
        curr_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select one piece', search_path,
                                                             f"(*.{extension})",
                                                             options=QFileDialog.DontUseNativeDialog)

        self.set_path(curr_path, extension)

    def set_path(self, path, extension):
        if extension == "wav":
            self.audio_path = path
        elif extension == "npz":
            self.score_path = path
        elif extension == "pt":
            self.model_path = path
