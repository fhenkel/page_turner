import os

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QFileDialog
from page_turner.config import DEFAULT_MODEL, DEFAULT_DIR


class DialogWindow(QDialog):
    def __init__(self):
        """
        This function initializes the dialog window for the page turner application
        """
        super().__init__()

        self.setWindowTitle("Please choose your setting")

        self.audio_path = None
        self.score_path = None
        self.n_pages = 0
        self.model_path = DEFAULT_MODEL

        font = QtGui.QFont()
        font.setBold(True)

        # setup audio selection
        audio_label = QtWidgets.QLabel("Audio", self)
        audio_label.setFont(font)

        audio_dropdown = QtWidgets.QComboBox()
        audio_dropdown.addItem("live")
        audio_dropdown.addItem("choose from library")
        audio_dropdown.activated[str].connect(self.react_to_audio_dropdown)

        # setup score selection
        score_label = QtWidgets.QLabel("Score", self)
        score_label.setFont(font)
        score_dropdown = QtWidgets.QComboBox()
        score_dropdown.addItem("live")
        score_dropdown.addItem("choose from library")
        score_dropdown.activated[str].connect(self.react_to_score_dropdown)

        # setup model selection
        model_label = QtWidgets.QLabel("Model", self)
        model_label.setFont(font)
        model_dropdown = QtWidgets.QComboBox()
        model_dropdown.addItem("default")
        model_dropdown.addItem("choose local")
        model_dropdown.activated[str].connect(self.react_to_model_dropdown)

        n_pages_label = QtWidgets.QLabel("#Pages", self)
        n_pages_label.setFont(font)
        self.n_pages_spinbox = QtWidgets.QSpinBox()
        self.n_pages_spinbox.setMinimum(0)
        self.n_pages_spinbox.valueChanged.connect(self.react_to_n_pages_change)

        # setup ok button
        ok_button = QtWidgets.QPushButton("ok")
        ok_button.clicked.connect(self.close)

        # setup layout
        grid_layout = QtWidgets.QGridLayout()

        i = 0
        grid_layout.addWidget(audio_label, i, 0)
        grid_layout.addWidget(audio_dropdown, i, 1)
        i += 1

        grid_layout.addWidget(score_label, i, 0)
        grid_layout.addWidget(score_dropdown, i, 1)
        i += 1

        grid_layout.addWidget(n_pages_label, i, 0)
        grid_layout.addWidget(self.n_pages_spinbox, i, 1)
        i += 1

        grid_layout.addWidget(model_label, i, 0)
        grid_layout.addWidget(model_dropdown, i, 1)
        i += 1

        grid_layout.addWidget(ok_button, i, 1)
        self.setLayout(grid_layout)

    def react_to_n_pages_change(self):
        self.n_pages = self.n_pages_spinbox.value()

    def react_to_audio_dropdown(self, text):
        """
        This function opens a dialog window to choose
        a wav file for the audio, when "choose from library" is clicked
        in the dropdown
        :param text: "live" or "choose from library"
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
        """

        if text == "choose local":
            self.choose_piece_dir("pt", os.path.join('..', 'models'))
        elif text == "default":
            self.model_path = DEFAULT_MODEL
        else:
            raise NotImplementedError

    def choose_piece_dir(self, extension, path=None):
        """
        This function opens a dialog window to choose a file
        with a specific extension in the default path or in another chosen path.
        The full path of the chosen file is saved to the respective variable.
        :param extension: str of the extension the file must have options: ["wav", "npz", "pt"]
        :param path: path to open first in the dialog
        """

        if path is not None:
            search_path = path
        else:
            search_path = DEFAULT_DIR
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
