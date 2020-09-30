import os
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data
from torch.optim import SGD

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix

from model.refinenet import Segmentor
from dataset import TrainDataset

from loss import CrossEntropyLoss2d, BCEWithLogitsLoss2d

from metric import evaluate

import settings


def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def lr_poly_scheduler(optim, init_lr, lr_decay_iter, iter, max_iter, poly_power):
    if iter % lr_decay_iter or iter > max_iter:
        return

    # calculate new lr
    new_lr = init_lr * (1 - float(iter) / max_iter) ** poly_power

    # set optim lr
    optim.param_groups[0]['lr'] = new_lr
    optim.param_groups[1]['lr'] = new_lr * 10


def main():

    # tensorboard writer
    writer = SummaryWriter(settings.TENSORBOARD_DIR)
    # makedir snapshot
    makedir(settings.CHECKPOINT_DIR)

    # enable cudnn 
    torch.backends.cudnn.enabled = True

    # create segmentor network
    model = Segmentor(pretrained=settings.PRETRAINED, num_classes=settings.NUM_CLASSES, 
                    modality=settings.MODALITY)
    
    model.train()
    model.cuda()

    torch.backends.cudnn.benchmark = True

    # dataset and dataloader
    dataset = TrainDataset()
    dataloader = data.DataLoader(dataset, batch_size=settings.BATCH_SIZE, 
                                shuffle=True, num_workers=settings.NUM_WORKERS,
                                pin_memory=True, drop_last=True)

    dataloader_iter = enumerate(dataloader)

    # optimizer for generator network (segmentor)
    optim = SGD(model.optim_parameters(settings.LR), lr=settings.LR, 
                        momentum=settings.LR_MOMENTUM, weight_decay=settings.WEIGHT_DECAY)


    # losses
    ce_loss = CrossEntropyLoss2d(ignore_index=settings.IGNORE_LABEL) # to use for segmentor

    # upsampling for the network output
    upsample = nn.Upsample(size=(settings.CROP_SIZE, settings.CROP_SIZE), mode='bilinear', align_corners=True)
    
    # confusion matrix ; to track metrics such as mIoU during training
    conf_mat = np.zeros((settings.NUM_CLASSES, settings.NUM_CLASSES))

    for i_iter in range(settings.MAX_ITER):

        # initialize losses
        loss_G_seg_value = 0

        # clear optim gradients and adjust learning rates
        optim.zero_grad()

        lr_poly_scheduler(optim, settings.LR, settings.LR_DECAY_ITER, 
                        i_iter, settings.MAX_ITER, settings.LR_POLY_POWER)

        
        ####### train generator #######

        # get the batch of data
        try:
            _, batch = next(dataloader_iter)
        except:
            dataloader_iter = enumerate(dataloader)
            _, batch = next(dataloader_iter)

        images, depths, labels = batch
        images = images.cuda()
        depths = depths.cuda()
        labels = labels.cuda()
        
        # get a mask where an elemnt is True for every pixel with ignore_label value
        ignore_mask = (labels == settings.IGNORE_LABEL)
        target_mask = torch.logical_not(ignore_mask)
        target_mask = target_mask.unsqueeze(dim=1)

        # get the output of generator
        if settings.MODALITY == 'rgb':
            predict = upsample(model(images))
        elif settings.MODALITY == 'middle':
            predict = upsample(model(images, depths))

        # calculate cross-entropy loss
        loss_G_seg = ce_loss(predict, labels)

        # accumulate loss, backward and store value
        loss_G_seg.backward()

        loss_G_seg_value += loss_G_seg.data.cpu().numpy()
        ####### end of train generator #######



        optim.step()

        # get pred and gt to compute confusion matrix
        seg_pred = np.argmax(predict.detach().cpu().numpy(), axis=1)
        seg_gt = labels.cpu().numpy().copy()

        seg_pred = seg_pred[target_mask.squeeze(dim=1).cpu().numpy()]
        seg_gt = seg_gt[target_mask.squeeze(dim=1).cpu().numpy()]

        conf_mat += confusion_matrix(seg_gt, seg_pred, labels=np.arange(settings.NUM_CLASSES))

        ####### log ########
        if i_iter % ((settings.TRAIN_SIZE // settings.BATCH_SIZE)) == 0 and i_iter != 0:
            metrics = evaluate(conf_mat)
            writer.add_scalar('Pixel Accuracy/Train', metrics['pAcc'], i_iter)
            writer.add_scalar('Mean Accuracy/Train', metrics['mAcc'], i_iter)
            writer.add_scalar('mIoU/Train', metrics['mIoU'], i_iter)
            writer.add_scalar('fwavacc/Train', metrics['fIoU'], i_iter)
            conf_mat = np.zeros_like(conf_mat)

        writer.add_scalar('Loss_G_SEG/Train', loss_G_seg_value, i_iter)
        writer.add_scalar('learning_rate_G/Train', optim.param_groups[0]['lr'], i_iter)


        print(  "iter = {:6d}/{:6d},\t loss_seg = {:.3f}".format(
                i_iter, settings.MAX_ITER,
                loss_G_seg_value))

        
        with open(settings.LOG_FILE, "a") as f:
            output_log = '{:6d},\t {:.8f}\n'.format(i_iter, loss_G_seg_value)
            f.write(output_log)

        # taking snapshot
        if i_iter >= settings.MAX_ITER:
            print('saving the final model ...')
            torch.save(model.state_dict(),osp.join(settings.CHECKPOINT_DIR, 'CHECKPOINT_'+str(settings.MAX_ITER)+'.pt'))
            break

        if i_iter % settings.SAVE_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(),osp.join(settings.CHECKPOINT_DIR, 'CHECKPOINT_'+str(i_iter)+'.pt'))
        
        
if __name__ == "__main__":
    main()








        




