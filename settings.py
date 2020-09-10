import numpy as np
from torch import Tensor


# Dataset properties
DATASET = 'nyu'

if DATASET == 'sun':
    NUM_CLASSES = 38
    DATA_ROOT = 'datasets/SUN_480x640_png_tiff/train/'
    DATA_LIST = 'datasets/SUN_480x640_png_tiff/train/train.txt'
    TRAIN_SIZE = 5285

elif DATASET == 'nyu':
    NUM_CLASSES = 41
    DATA_ROOT = 'datasets/NYU_V2/train/'
    DATA_LIST = 'datasets/NYU_V2/train/train.txt'
    TRAIN_SIZE = 795

# Data settings
IMG_MEAN = Tensor(np.array([0.485, 0.456, 0.406]))
IMG_STD = Tensor(np.array([0.229, 0.224, 0.225]))
DPT_MEAN = (0.423,)
DPT_STD = (0.272,)
SCALE_RANGE = (0.5, 2.0)
CROP_SIZE = 350
IGNORE_LABEL = 0
DPT_IGNORE_LABEL = 0
BCE_IGNORE_LABEL = -2

# Model settings
PRETRAINED = True
MODALITY = 'rgb'

# Train settings
DEVICE = 'gpu'
LOG_DIR = 'gan1_rgb'
SAVE_EVERY = 10000

MAX_ITER = 70000


LR = 5e-4
LR_D = 1e-4
LR_MOMENTUM = 0.9
LR_DECAY_ITER = 10
LR_POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4

LAMBDA_ADV_SEG = 1e-2

BATCH_SIZE = 5
NUM_WORKERS = 16


