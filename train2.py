import os
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torch.utils.data as data
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


def save_metrics(conf_mat, writer, epoch):
    metrics = evaluate(conf_mat)
    writer.add_scalar('Pixel Accuracy/Train', metrics['pAcc'], epoch)
    writer.add_scalar('Mean Accuracy/Train', metrics['mAcc'], epoch)
    writer.add_scalar('mIoU/Train', metrics['mIoU'], epoch)
    writer.add_scalar('fwavacc/Train', metrics['fIoU'], epoch)


def train_one_epoch(model, optimizer, dataloader, epoch, upsample, ce_loss, writer, print_freq=10):

    # confusion matrix ; to track metrics such as mIoU during training
    conf_mat = np.zeros((settings.NUM_CLASSES, settings.NUM_CLASSES))

    data_size = len(dataloader)

    # initialize losses
    loss_G_seg_values = []

    for i_iter, batch in enumerate(dataloader):

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

        
        optimizer.zero_grad()
        loss_G_seg.backward()
        optimizer.step()

        loss_G_seg_values.append(loss_G_seg.data.cpu().numpy())

        # # get pred and gt to compute confusion matrix
        # seg_pred = np.argmax(predict.detach().cpu().numpy(), axis=1)
        # seg_gt = labels.cpu().numpy().copy()

        # seg_pred = seg_pred[target_mask.squeeze(dim=1).cpu().numpy()]
        # seg_gt = seg_gt[target_mask.squeeze(dim=1).cpu().numpy()]

        # conf_mat += confusion_matrix(seg_gt, seg_pred, labels=np.arange(settings.NUM_CLASSES))

        # with open(settings.LOG_FILE, "a") as f:
        #     output_log = '{}, {},\t {:.8f}\n'.format(epoch, i_iter, loss_G_seg_value)
        #     f.write(output_log)

        
        if i_iter % print_freq == 0 and i_iter != 0:
            loss_G_seg_value = np.mean(loss_G_seg_values)
            loss_G_seg_values = []
            writer.add_scalar('Loss_G_SEG/Train', loss_G_seg_value, i_iter+epoch*data_size)
            writer.add_scalar('learning_rate_G/Train', optimizer.param_groups[0]['lr'], i_iter+epoch*data_size)

            print("epoch = {:3d}/{:3d}: iter = {:3d},\t loss_seg = {:.3f}".format(
                epoch, settings.EPOCHS, i_iter, loss_G_seg_value))

    # save_metrics(conf_mat, writer, epoch)

    

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
    optimizer = optim.SGD(model.optim_parameters(settings.LR), lr=settings.LR, 
                        momentum=settings.LR_MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

    # lr scheduler for optimizer
    lr_lambda = lambda epoch: (1 - epoch / settings.EPOCHS) ** settings.LR_POLY_POWER
    lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)


    # losses
    ce_loss = CrossEntropyLoss2d(ignore_index=settings.IGNORE_LABEL) # to use for segmentor

    # upsampling for the network output
    upsample = nn.Upsample(size=(settings.CROP_SIZE, settings.CROP_SIZE), mode='bilinear', align_corners=True)
    
    last_epoch = -1
    if settings.RESUME_TRAIN:
        checkpoint = torch.load(settings.LAST_CHECKPOINT)

        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    for epoch in range(last_epoch+1, settings.EPOCHS):

        train_one_epoch(model, optimizer, dataloader, epoch, 
                        upsample, ce_loss, writer, print_freq=5)

        lr_scheduler.step()

        if epoch % settings.CHECKPOINT_FREQ == 0 and epoch != 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }
            print('saving a checkpoint in epoch {}'.format(epoch))
            torch.save(checkpoint, osp.join(settings.CHECKPOINT_DIR, 'CHECKPOINT_'+str(epoch)+'.tar'))

        # save the final model
        if epoch >= settings.EPOCHS-1:
            print('saving the final model')
            torch.save(checkpoint, osp.join(settings.CHECKPOINT_DIR, 'CHECKPOINT_'+str(epoch+1)+'.tar'))

        


        
        
if __name__ == "__main__":
    main()

