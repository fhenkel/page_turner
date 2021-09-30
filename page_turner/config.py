import os

SCORE_HEIGHT = 1181
SCORE_WIDTH = 835
SCALE_WIDTH = 416
SCALE_FACTOR = SCORE_HEIGHT / SCALE_WIDTH
PADDING = (SCORE_HEIGHT - SCORE_WIDTH) // 2

DEFAULT_MODEL = os.path.join('..', 'models', 'eop_model', 'best_model.pt')
