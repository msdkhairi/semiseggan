import numpy as np
from torch import Tensor


# Dataset properties
DATASET = 'sun'

if DATASET == 'sun':
    NUM_CLASSES = 38
    DATA_ROOT = 'drive/MyDrive/datasets/sunrgbd/train/'
    DATA_LIST = 'drive/MyDrive/datasets/sunrgbd/train/train.txt'
    DATA_ROOT_VAL = 'drive/MyDrive/datasets/sunrgbd/val/'
    DATA_LIST_VAL = 'drive/MyDrive/datasets/sunrgbd/val/val.txt'
    TRAIN_SIZE = 5285

elif DATASET == 'nyu':
    NUM_CLASSES = 41
    DATA_ROOT = 'drive/MyDrive/datasets/nyuv2/train/'
    DATA_LIST = 'drive/MyDrive/datasets/nyuv2/train/train.txt'
    DATA_ROOT_VAL = 'drive/MyDrive/datasets/nyuv2/val/'
    DATA_LIST_VAL = 'drive/MyDrive/datasets/nyuv2/val/val.txt'
    TRAIN_SIZE = 795

# Data settings
IMG_MEAN = Tensor(np.array([0.485, 0.456, 0.406]))
IMG_STD = Tensor(np.array([0.229, 0.224, 0.225]))
# NYU depth 
# DPT_MEAN = (0.423,)
# DPT_STD = (0.272,)

# SUN depth
DPT_MEAN = (0.388,)
DPT_STD = (0.200,)

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
LOG_DIR = 'refinenet50_rgb'
TENSORBOARD_DIR = LOG_DIR_ROOT + 'runs/' + DATASET + '/'+ LOG_DIR
CHECKPOINT_DIR = LOG_DIR_ROOT + 'checkpoints_' + DATASET + '_' + LOG_DIR
LOG_FILE = LOG_DIR_ROOT + 'log_' + DATASET + '_' + LOG_DIR + '.txt'


# EVAL and RESUME
RESUME_TRAIN = False
LAST_CHECKPOINT = CHECKPOINT_DIR + '/CHECKPOINT_500.tar'
OUTPUT_FILE = LOG_DIR_ROOT + 'results_' + DATASET + '_' + LOG_DIR + '.txt'
OUTPUT_DIR = LOG_DIR_ROOT + 'results_' + DATASET + '_' + LOG_DIR + '/'



# Train settings
MANUAL_SEED = 27

DEVICE = 'gpu'
CHECKPOINT_FREQ = 25
EPOCHS = 500
EVAL_FREQ = 10 # number of epochs where model is evaluated on train data and validation data


LR = 5e-4
LR_D = 1e-4
LR_MOMENTUM = 0.9
LR_DECAY_ITER = 10
LR_POLY_POWER = 0.9
LR_POLY_STEP = 10
WEIGHT_DECAY = 1e-4


LAMBDA_ADV_SEG = 1e-2

BATCH_SIZE = 8
NUM_WORKERS = 16



