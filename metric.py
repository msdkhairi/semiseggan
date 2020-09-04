import numpy as np

from sklearn.metrics import confusion_matrix

import settings

def evaluate(conf_mat):


    ignore_label = settings.IGNORE_LABEL
    num_classes = settings.NUM_CLASSES

    # omit ignore label row and column from confusion matrix 
    if ignore_label >= 0 and ignore_label < num_classes:
        row_omitted = np.delete(conf_mat, ignore_label, axis=0)
        conf_mat = np.delete(row_omitted, ignore_label, axis=1)

    # calculate metrics
    acc = np.diag(conf_mat).sum() / conf_mat.sum()
    acc_cls = np.diag(conf_mat) / conf_mat.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iou = np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat))
    mean_iou = np.nanmean(iou)
    freq = conf_mat.sum(axis=1) / conf_mat.sum()
    fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

    return {
        'pAcc' : acc,
        'mAcc': acc_cls,
        'iou' : iou,
        'fIoU': fwavacc,
        'mIoU': mean_iou,
    }

