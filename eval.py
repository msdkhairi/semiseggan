import os

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data

from PIL import Image

from sklearn.metrics import confusion_matrix

from model.refinenet import Segmentor
from dataset import TestDataset

from metric import evaluate_conf_mat

import settings


def get_palette(num_classes):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """

    n = num_classes
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def write_to_log(metrics, print_output=False):
    with open(settings.OUTPUT_FILE, 'w') as f:
        f.write('Semantic Segmentation Results on {} with model {}\n\n'.format(
            settings.DATASET, settings.LAST_CHECKPOINT))
        for key, value in metrics.items():
            if key == 'iou':
                output = '{}:\n{}\n'.format(key, value)
            else:
                output = '{}: {}\n'.format(key, value)
            f.write(output)
            if print_output:
                print(output, end='')


def evaluate(model, dataloader):

    model.eval()
    model.cuda()

    upsample = nn.Upsample((427, 561), mode='bilinear', align_corners=True)

    conf_mat = np.zeros((settings.NUM_CLASSES, settings.NUM_CLASSES))

    for i_iter, batch in enumerate(dataloader):
        images, depths, labels = batch
        images = images.cuda()
        depths = depths.cuda()
        labels = labels.cuda()

        predict = None
        with torch.no_grad():
            if settings.MODALITY == 'rgb':
                predict = upsample(model(images))
            elif settings.MODALITY == 'middle':
                predict = upsample(model(images, depths))

        seg_pred = np.argmax(predict[0].cpu().numpy(), axis=0)
        seg_gt = labels[0].cpu().numpy()

        conf_mat += confusion_matrix(seg_gt.reshape(-1), seg_pred.reshape(-1), labels=np.arange(settings.NUM_CLASSES))

        if i_iter % 100 == 0:
           print('{} val images processed'.format(i_iter))

    return conf_mat

def main():
    if not os.path.exists(settings.OUTPUT_DIR):
        os.makedirs(settings.OUTPUT_DIR)

    checkpoint = torch.load(settings.LAST_CHECKPOINT)

    model = Segmentor(num_classes=settings.NUM_CLASSES, modality=settings.MODALITY)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    model.cuda()

    dataset = TestDataset(data_root=settings.DATA_ROOT_VAL, data_list=settings.DATA_LIST_VAL)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, 
                                num_workers=settings.NUM_WORKERS, pin_memory=True)
    dataloader_iter = enumerate(dataloader)

    upsample = nn.Upsample((427, 561), mode='bilinear', align_corners=True)

    conf_mat = np.zeros((settings.NUM_CLASSES, settings.NUM_CLASSES))

    palette = get_palette(settings.NUM_CLASSES)

    out_dir = settings.OUTPUT_DIR

    for i_iter in range(len(dataloader)):
        
        _, batch = next(dataloader_iter)
        images, depths, labels = batch
        images = images.cuda()
        depths = depths.cuda()
        labels = labels.cuda()

        # get the output of model
        with torch.no_grad():
            if settings.MODALITY == 'rgb':
                predict = upsample(model(images))
            elif settings.MODALITY == 'middle':
                predict = upsample(model(images, depths))
    
        seg_pred = np.argmax(predict[0].cpu().numpy(), axis=0)
        seg_gt = labels[0].cpu().numpy()

        conf_mat += confusion_matrix(seg_gt.reshape(-1), seg_pred.reshape(-1), labels=np.arange(settings.NUM_CLASSES))

        seg_pred = Image.fromarray(seg_pred.astype(np.uint8))
        seg_gt = Image.fromarray(seg_gt.astype(np.uint8))
        
        seg_pred.putpalette(palette)
        seg_gt.putpalette(palette)
        
        seg_pred_fn = out_dir + str(i_iter) + '_pred.png'
        seg_gt_fn = out_dir + str(i_iter) + '_gt.png'

        seg_pred.save(seg_pred_fn)
        seg_gt.save(seg_gt_fn)

        if i_iter % 100 == 0:
           print('{} images processed' .format(i_iter))


    metrics = evaluate_conf_mat(conf_mat)
    write_to_log(metrics, print_output=True)


if __name__ == '__main__':
    main()