import os
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data
from torch import optim

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix

from model.refinenet import Segmentor
from model.discriminator import Discriminator
from dataset import TrainDataset, TestDataset

from loss import CrossEntropyLoss2d, BCEWithLogitsLoss2d, FocalLoss

from metric import evaluate_conf_mat
from eval import evaluate

from utils import makedir, save_metrics

import settings


def make_D_label(label, D_output):
    D_label = torch.ones_like(D_output) * label
    D_label = D_label.clone().detach().requires_grad_(True).cuda()
    return D_label



def save_checkpoint(epoch, model_G, model_D, optim_G, optim_D, lr_scheduler_G, lr_scheduler_D):
    checkpoint = {
        'epoch': epoch,
        'model_G_state_dict': model_G.state_dict(),
        'model_D_state_dict': model_D.state_dict(),
        'optim_G_state_dict': optim_G.state_dict(),
        'optim_D_state_dict': optim_D.state_dict(),
        'lr_scheduler_G_state_dict': lr_scheduler_G.state_dict(),
        'lr_scheduler_D_state_dict': lr_scheduler_D.state_dict()
        }
    
    print('saving a checkpoint in epoch {}'.format(epoch))
    torch.save(checkpoint, osp.join(settings.CHECKPOINT_DIR, 'CHECKPOINT_'+str(epoch)+'.tar'))



def train_one_epoch(model_G, model_D, optim_G, optim_D, dataloader, test_dataloader, epoch, 
                        upsample, ce_loss, bce_loss, writer, print_freq=5, eval_freq=settings.EVAL_FREQ):
    
    max_iter = len(dataloader)
            
    # initialize losses
    loss_G_seg_values = []
    loss_adv_seg_values = []
    loss_D_values = []

    eval_trainval = False
    if epoch % eval_freq == 0:
        eval_trainval = True

    # confusion matrix ; to track metrics such as mIoU during training
    conf_mat = np.zeros((settings.NUM_CLASSES, settings.NUM_CLASSES))

    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    for i_iter, batch in enumerate(dataloader):

        images, depths, labels = batch
        images = images.cuda()
        depths = depths.cuda()
        labels = labels.cuda()

        optim_G.zero_grad()
        optim_D.zero_grad()

        ####### train generator #######
        # disable accumulating grads in discriminator
        for param in model_D.parameters():
            param.requires_grad = False

        # get a mask where an elemnt is True for every pixel with ignore_label value
        ignore_mask = (labels == settings.IGNORE_LABEL)
        target_mask = torch.logical_not(ignore_mask)
        target_mask = target_mask.unsqueeze(dim=1)

        # get the output of generator
        if settings.MODALITY == 'rgb':
            predict = upsample(model_G(images))
        elif settings.MODALITY == 'middle':
            predict = upsample(model_G(images, depths))

        # calculate cross-entropy loss
        loss_G_seg = ce_loss(predict, labels)

        # calculate adversarial loss
        D_output = upsample(model_D(F.softmax(predict, dim=1)))
        loss_adv = bce_loss(D_output, make_D_label(gt_label, D_output), target_mask)

        # accumulate loss, backward and store value
        loss_G = loss_G_seg + settings.LAMBDA_ADV_SEG * loss_adv
        loss_G.backward()

        loss_G_seg_values.append(loss_G_seg.data.cpu().numpy())
        loss_adv_seg_values.append(loss_adv.data.cpu().numpy())

        if eval_trainval:
            # get pred and gt to compute confusion matrix
            seg_pred = np.argmax(predict.detach().cpu().numpy(), axis=1)
            seg_gt = labels.cpu().numpy().copy()

            seg_pred = seg_pred[target_mask.squeeze(dim=1).cpu().numpy()]
            seg_gt = seg_gt[target_mask.squeeze(dim=1).cpu().numpy()]

            conf_mat += confusion_matrix(seg_gt, seg_pred, labels=np.arange(settings.NUM_CLASSES))

        ####### end of train generator #######


        ####### train discriminator #######

        # activate the gradient accumulation in D
        for param in model_D.parameters():
            param.requires_grad = True

        # detach from G
        predict = predict.detach()

        D_output = upsample(model_D(F.softmax(predict, dim=1)))
        loss_D = bce_loss(D_output, make_D_label(pred_label, D_output), target_mask)
        loss_D.backward()
        loss_D_values.append(loss_D.data.cpu().numpy())

        # pass ground truth to discriminator
        
        gt = labels.clone().detach().cuda()
        gt_one_hot = F.one_hot(gt, num_classes=settings.NUM_CLASSES).permute(0,3,1,2).contiguous().float()
        D_output = upsample(model_D(gt_one_hot))

        loss_D = bce_loss(D_output, make_D_label(gt_label, D_output), target_mask)
        loss_D.backward()
        loss_D_values.append(loss_D.data.cpu().numpy())
        ####### end of train discriminator #######

        optim_G.step()
        optim_D.step()

        if i_iter % print_freq == 0 and i_iter != 0:
            loss_G_seg_value = np.mean(loss_G_seg_values)
            loss_G_seg_values = []

            loss_adv_seg_value = np.mean(loss_adv_seg_values)
            loss_adv_seg_values = []

            loss_D_value = np.mean(loss_D_values)
            loss_D_values = []

            writer.add_scalar('Loss_G_SEG/Train', loss_G_seg_value, i_iter+epoch*max_iter)
            writer.add_scalar('Loss_G_SEG_ADV/Train', loss_adv_seg_value, i_iter+epoch*max_iter)
            writer.add_scalar('Loss_D/Train', loss_D_value, i_iter+epoch*max_iter)
            writer.add_scalar('learning_rate_G/Train', optim_G.param_groups[0]['lr'], i_iter+epoch*max_iter)
            writer.add_scalar('learning_rate_D/Train', optim_D.param_groups[0]['lr'], i_iter+epoch*max_iter)

            print("epoch = {:3d}/{:3d}: iter = {:3d},\t loss_seg = {:.3f},\t loss_adv = {:.3f},\t loss_d = {:.3f}".format(
                epoch, settings.EPOCHS, i_iter, loss_G_seg_value, loss_adv_seg_value, loss_D_value))

        
    if eval_trainval:
        save_metrics(conf_mat, writer, epoch*max_iter, 'Train')
        conf_mat = evaluate(model_G, test_dataloader)
        save_metrics(conf_mat, writer, epoch*max_iter, 'Val')
        model_G.train()


def main():

    # set torch and numpy seed for reproducibility
    torch.manual_seed(settings.MANUAL_SEED)
    torch.cuda.manual_seed_all(settings.MANUAL_SEED)
    np.random.seed(settings.MANUAL_SEED)

    # tensorboard writer
    writer = SummaryWriter(settings.TENSORBOARD_DIR)
    # makedir snapshot
    makedir(settings.CHECKPOINT_DIR)

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


    # dataset and dataloader
    dataset = TrainDataset()
    dataloader = data.DataLoader(dataset, batch_size=settings.BATCH_SIZE, 
                                shuffle=True, num_workers=settings.NUM_WORKERS,
                                pin_memory=True, drop_last=True)

    test_dataset = TestDataset(data_root=settings.DATA_ROOT_VAL, data_list=settings.DATA_LIST_VAL)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                num_workers=settings.NUM_WORKERS, pin_memory=True)


    # optimizer for generator network (segmentor)
    optim_G = optim.SGD(model_G.optim_parameters(settings.LR), lr=settings.LR, 
                        momentum=settings.LR_MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
    
    # lr scheduler for optimi_G
    # lr_lambda_G = lambda epoch: (1 - epoch / settings.EPOCHS) ** settings.LR_POLY_POWER
    # lr_scheduler_G = optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=lr_lambda_G)
    lr_scheduler_G = optim.lr_scheduler.StepLR(optim_G, step_size=settings.LR_POLY_STEP, gamma=0.9)

    # optimizer for discriminator network
    optim_D = optim.Adam(model_D.parameters(), settings.LR_D)

    # lr scheduler for optimi_D
    # lr_lambda_D = lambda epoch: (1 - epoch / settings.EPOCHS) ** settings.LR_POLY_POWER
    # lr_scheduler_D = optim.lr_scheduler.LambdaLR(optim_D, lr_lambda=lr_lambda_D)
    lr_scheduler_D = optim.lr_scheduler.StepLR(optim_D, step_size=settings.LR_POLY_STEP, gamma=0.9)

    # losses
    ce_loss = CrossEntropyLoss2d(ignore_index=settings.IGNORE_LABEL) # to use for segmentor
    bce_loss = BCEWithLogitsLoss2d() # to use for discriminator

    # upsampling for the network output
    upsample = nn.Upsample(size=(settings.CROP_SIZE, settings.CROP_SIZE), mode='bilinear', align_corners=True)
    
    # load the model to resume training
    last_epoch = -1
    if settings.RESUME_TRAIN:
        checkpoint = torch.load(settings.LAST_CHECKPOINT)

        model_G.load_state_dict(checkpoint['model_G_state_dict'])
        model_G.train()
        model_G.cuda()

        model_D.load_state_dict(checkpoint['model_D_state_dict'])
        model_D.train()
        model_D.cuda()

        optim_G.load_state_dict(checkpoint['optim_G_state_dict'])
        optim_D.load_state_dict(checkpoint['optim_D_state_dict'])

        lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G_state_dict'])
        lr_scheduler_D.load_state_dict(checkpoint['lr_scheduler_D_state_dict'])

        last_epoch = checkpoint['epoch']
        

        # purge the logs after the last_epoch
        writer = SummaryWriter(settings.TENSORBOARD_DIR, purge_step=(last_epoch+1)*len(dataloader))

    for epoch in range(last_epoch+1, settings.EPOCHS+1):

        train_one_epoch(model_G, model_D, optim_G, optim_D, dataloader, test_dataloader, epoch, 
                        upsample, ce_loss, bce_loss, writer, print_freq=5, eval_freq=settings.EVAL_FREQ)


        if epoch % settings.CHECKPOINT_FREQ == 0 and epoch != 0:
            save_checkpoint(epoch, model_G, model_D, optim_G, optim_D, 
                            lr_scheduler_G, lr_scheduler_D)

        # save the final model
        if epoch >= settings.EPOCHS:
            print('saving the final model')
            save_checkpoint(epoch, model_G, model_D, optim_G, optim_D, 
                            lr_scheduler_G, lr_scheduler_D)
            writer.close()

        lr_scheduler_G.step()
        lr_scheduler_D.step()
        

if __name__ == "__main__":
    main()


