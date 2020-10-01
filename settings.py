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
    DATA_ROOT = 'drive/My Drive/datasets/nyuv2/train/'
    DATA_LIST = 'drive/My Drive/datasets/nyuv2/train/train.txt'
    DATA_ROOT_VAL = 'drive/My Drive/datasets/nyuv2/val/'
    DATA_LIST_VAL = 'drive/My Drive/datasets/nyuv2/val/val.txt'
    TRAIN_SIZE = 795

# Data settings
IMG_MEAN = Tensor(np.array([0.485, 0.456, 0.406]))
IMG_STD = Tensor(np.array([0.229, 0.224, 0.225]))
DPT_MEAN = (0.423,)
DPT_STD = (0.272,)
SCALE_RANGE = (0.7, 2.0)
CROP_SIZE = 350
IGNORE_LABEL = 0
DPT_IGNORE_LABEL = 0
BCE_IGNORE_LABEL = -2

# Model settings
PRETRAINED = True
MODALITY = 'rgb'

# logging
LOG_DIR_ROOT = 'drive/My Drive/Logs/'
LOG_DIR = 'refienenet_rgb'
TENSORBOARD_DIR = LOG_DIR_ROOT + 'runs/' + DATASET + '/'+ LOG_DIR
CHECKPOINT_DIR = LOG_DIR_ROOT + 'checkpoints_' + DATASET + '_' + LOG_DIR
LOG_FILE = LOG_DIR_ROOT + 'log_' + DATASET + '_' + LOG_DIR + '.txt'


# 
RESUME_TRAIN = True
LAST_CHECKPOINT = LOG_DIR_ROOT + 'checkpoints_nyu_refinenet_rgb/CHECKPOINT_350.pth'
OUTPUT_FILE = LOG_DIR_ROOT + 'results_' + DATASET + '_' + LOG_DIR + '.txt'
OUTPUT_DIR = LOG_DIR_ROOT + 'results_' + DATASET + '_' + LOG_DIR + '/'



# Train settings
DEVICE = 'gpu'
SAVE_EVERY = 10000
CHECKPOINT_FREQ = 50

MAX_ITER = 70000
EPOCHS = 500


LR = 5e-4
LR_D = 1e-4
LR_MOMENTUM = 0.9
LR_DECAY_ITER = 10
LR_POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4

LAMBDA_ADV_SEG = 1e-2

BATCH_SIZE = 8
NUM_WORKERS = 16



