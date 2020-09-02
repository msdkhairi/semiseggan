import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from model.refinenet import Segmentor
from model.discriminator import Discriminator
from dataset import TrainDataset

from loss import CrossEntropyLoss2d, BCEWithLogitsLoss2d

import settings


def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def lr_poly_scheduler(optim_G, optim_D, init_lr, lr_decay_iter, iter, max_iter, poly_power):
    if iter % lr_decay_iter or iter > max_iter:
        return

    # calculate new lr
    new_lr = init_lr * (1 - float(iter) / max_iter) ** poly_power

    # set optim_G lr
    optim_G.param_groups[0]['lr'] = new_lr
    optim_G.param_groups[1]['lr'] = new_lr * 10

    # set optim_D lr
    optim_D.param_groups[0]['lr'] = new_lr


def make_D_label(label, D_output):
    D_label = np.ones_like(D_output) * label
    D_label = torch.tensor(D_label, dtype=torch.float64, requires_grad=True).cuda()
    return D_label


def make_D_label2(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = (np.ones(ignore_mask.shape)*label)
    D_label[ignore_mask] = settings.BCE_IGNORE_LABEL
    # D_label = Variable(torch.FloatTensor(D_label)).cuda()
    D_label = torch.tensor(D_label, dtype=torch.float64, requires_grad=True).cuda()
    return D_label


def main():

    # tensorboard writer
    write_dir = "runs/" + settings.DATASET + "/"+ settings.LOG_DIR
    writer = SummaryWriter(write_dir)

    # enable cudnn 
    torch.backends.cudnn.enabled = True

    # create segmentor network
    model_G = Segmentor(pretrained=settings.PRETRAINED, num_classes=settings.NUM_CLASSES, 
                    modality=settings.MODALITY)
    
    model_G.train()
    model_G.cuda()

    torch.backends.cudnn.benchmark = True

    # create discriminator network
    model_D = Discriminator(settings.NUM_CLASSES)
    model_D.train()
    model_D.cuda()

    # snapshot
    snapshot_dir = './snapshots_' + settings.DATASET + "_" + settings.LOG_DIR + "/"
    makedir(snapshot_dir)

    # dataset and dataloader
    dataset = TrainDataset()
    dataloader = data.DataLoader(dataset, batch_size=settings.BATCH_SIZE, 
                                shuffle=True, num_workers=settings.NUM_WORKERS,
                                pin_memory=True, drop_last=True)

    dataloader_iter = enumerate(dataloader)

    # optimizer for generator network (segmentor)
    optim_G = optim.SGD(model_G.optim_parameters(settings.LR), lr=settings.LR, 
                        momentum=settings.LR_MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

    # optimizer for discriminator network
    optim_D = optim.Adam(model_D.parameters(), settings.LR)

    # losses
    ce_loss = CrossEntropyLoss2d(ignore_index=settings.IGNORE_LABEL) # to use for segmentor
    bce_loss = BCEWithLogitsLoss2d() # to use for discriminator

    # upsampling for the network output
    upsample = nn.Upsample(size=(settings.CROP_SIZE, settings.CROP_SIZE), mode='bilinear', align_corners=True)

    # labels for adversarial training
    pred_label = 0
    gt_label = 1
    
    # confusion matrix ; to track metrics such as mIoU during training
    conf_mat = np.zeros((settings.NUM_CLASSES, settings.NUM_CLASSES))

    for i_iter in range(settings.MAX_ITER):

        # initialize losses
        loss_G_seg_value = 0
        loss_adv_value = 0
        loss_D_value = 0

        # clear optim gradients and adjust learning rates
        optim_G.zero_grad()
        optim_D.zero_grad()

        lr_poly_scheduler(optim_G, optim_D, settings.LR, settings.LR_DECAY_ITER, 
                        i_iter, settings.MAX_ITER, settings.LR_POLY_POWER)

        
        ####### train generator #######

        # not accumulate grads in discriminator
        for param in model_D.parameters():
            param.requires_grad = False

        # get the batch of data
        _, batch = next(dataloader_iter)

        images, depths, labels = batch
        
        # get a mask where is True for every pixel with ignore_label value
        ignore_mask = (labels.numpy() == settings.IGNORE_LABEL)

        # get the output of generator
        if settings.MODALITY == 'rgb':
            predict = upsample(model_G(images))
        elif settings.MODALITY == 'middle':
            predict = upsample(model_G(images, depths))

        # calculate cross-entropy loss
        loss_G_seg = ce_loss(predict, labels)

        # calculate adversarial loss
        D_output = upsample(model_D(F.softmax(predict, dim=1)))
        loss_adv = bce_loss(D_output, make_D_label(gt_label, D_output), np.logical_not(ignore_mask))

        # accumulate loss, backward and store value
        loss = loss_G_seg + settings.LAMBDA_ADV_SEG * loss_adv
        loss.backward()

        loss_G_seg_value += loss_G_seg.cpu().numpy()
        loss_adv_value += loss_adv.cpu().numpy()

        ####### end of train generator #######


        ####### train discriminator #######

        # reset the gradient accumulation
        for param in model_D.parameters():
            param.requires_grad = True

        predict = predict.detach()
        

        ####### end of train discriminator #######



        




