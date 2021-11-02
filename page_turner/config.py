import os

SCORE_HEIGHT = 1181
SCORE_WIDTH = 835
SCALE_WIDTH = 416
SCALE_FACTOR = SCORE_HEIGHT / SCALE_WIDTH
SPEC_VIS_WINDOW = 400
PADDING = (SCORE_HEIGHT - SCORE_WIDTH) // 2

DEFAULT_MODEL = os.path.join('..', 'models', 'test_model', 'best_model.pt')
DEFAULT_DIR = os.path.join('..', 'demo_piece')

