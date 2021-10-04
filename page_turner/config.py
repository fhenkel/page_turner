import os

SCORE_HEIGHT = 1181
SCORE_WIDTH = 835
SCALE_WIDTH = 416
SCALE_FACTOR = SCORE_HEIGHT / SCALE_WIDTH
PADDING = (SCORE_HEIGHT - SCORE_WIDTH) // 2

# DEFAULT_MODEL = os.path.join('..', 'models', 'eop_model', 'best_model.pt')
DEFAULT_MODEL = os.path.join('..', 'models', 'model', 'best_model.pt')
# DEFAULT_DIR = os.path.join('..', 'demo_piece')
DEFAULT_DIR = "/home/florian/mounts/rk7/home/florianh/frontiers_data/msmd_22050/msmd_test"
