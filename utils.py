import os
import os.path as osp

import torch

from metric import evaluate_conf_mat 

def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_metrics(conf_mat, writer, step, mode='Train'):
    metrics = evaluate_conf_mat(conf_mat)
    writer.add_scalar('Pixel Accuracy/{}'.format(mode), metrics['pAcc'], step)
    writer.add_scalar('Mean Accuracy/{}'.format(mode), metrics['mAcc'], step)
    writer.add_scalar('Mean IoU/{}'.format(mode), metrics['mIoU'], step)
    writer.add_scalar('fwavacc/{}'.format(mode), metrics['fIoU'], step)

