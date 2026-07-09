"""
config.py
---------
One place for all the knobs. I pulled these out of train.py so the notebook and
the scripts share the exact same numbers.
"""

import torch

# data
NUM_PAIRS = 5000
MAX_LEN = 20
MIN_FREQ = 1

# model
EMB_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.2

# training
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
CLIP = 1.0
TEACHER_FORCING_RATIO = 0.5

# misc
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "translator.pt"
