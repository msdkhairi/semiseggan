import os
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix

from model.refinenet import Segmentor
from model.discriminator import Discriminator
from dataset import TrainDataset

from loss import CrossEntropyLoss2d, BCEWithLogitsLoss2d

from metric import evaluate

import settings


def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def lr_poly_scheduler(optim_G, optim_D, init_lr, init_lr_D, lr_decay_iter, iter, max_iter, poly_power):
    if iter % lr_decay_iter or iter > max_iter:
        return

    # calculate new lr
    new_lr = init_lr * (1 - float(iter) / max_iter) ** poly_power
    new_lr_D = init_lr_D * (1 - float(iter) / max_iter) ** poly_power

    # set optim_G lr
    optim_G.param_groups[0]['lr'] = new_lr
    optim_G.param_groups[1]['lr'] = new_lr * 10

    # set optim_D lr
    optim_D.param_groups[0]['lr'] = new_lr_D


def make_D_label(label, D_output):
    D_label = torch.ones_like(D_output) * label
    D_label = D_label.clone().detach().requires_grad_(True).cuda()
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

    # snapshot and logg
    snapshot_dir = './snapshots_' + settings.DATASET + "_" + settings.LOG_DIR + "/"
    makedir(snapshot_dir)
    log_file = "log_" + settings.DATASET + "_" + settings.LOG_DIR + ".txt"

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
    optim_D = optim.Adam(model_D.parameters(), settings.LR_D)

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
        loss_adv_seg_value = 0
        loss_D_value = 0

        # clear optim gradients and adjust learning rates
        optim_G.zero_grad()
        optim_D.zero_grad()

        lr_poly_scheduler(optim_G, optim_D, settings.LR, settings.LR_D, settings.LR_DECAY_ITER, 
                        i_iter, settings.MAX_ITER, settings.LR_POLY_POWER)

        
        ####### train generator #######

        # not accumulate grads in discriminator
        for param in model_D.parameters():
            param.requires_grad = False

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
        
        # get a mask where is True for every pixel with ignore_label value
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
        loss = loss_G_seg + settings.LAMBDA_ADV_SEG * loss_adv
        loss.backward()

        loss_G_seg_value += loss_G_seg.data.cpu().numpy()
        loss_adv_seg_value += loss_adv.data.cpu().numpy()

        ####### end of train generator #######


        ####### train discriminator #######

        # pass prediction to discriminator

        # reset the gradient accumulation
        for param in model_D.parameters():
            param.requires_grad = True

        # detach from G
        predict = predict.detach()
        
        D_output = upsample(model_D(F.softmax(predict, dim=1)))
        loss_D = bce_loss(D_output, make_D_label(pred_label, D_output), target_mask)
        loss_D.backward()
        loss_D_value += loss_D.data.cpu().numpy()

        # pass ground truth to discriminator
        
        gt = labels.clone().detach().cuda()
        gt_one_hot = F.one_hot(gt, num_classes=settings.NUM_CLASSES).permute(0,3,1,2).float()
        D_output = upsample(model_D(gt_one_hot))

        loss_D = bce_loss(D_output, make_D_label(gt_label, D_output), target_mask)
        loss_D.backward()
        loss_D_value += loss_D.data.cpu().numpy()

        ####### end of train discriminator #######

        optim_G.step()
        optim_D.step()

        # get pred and gt to compute confusion matrix
        seg_pred = np.argmax(predict.cpu().numpy(), axis=1)
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
        writer.add_scalar('Loss_D/Train', loss_D_value, i_iter)
        writer.add_scalar('Loss_G_SEG_adv/Train', loss_adv_seg_value, i_iter)
        writer.add_scalar('learning_rate_G/Train', optim_G.param_groups[0]['lr'], i_iter)
        writer.add_scalar('learning_rate_D/Train', optim_D.param_groups[0]['lr'], i_iter)


        print(  "iter = {:6d}/{:6d},\t loss_seg = {:.3f}, loss_adv = {:.3f}, loss_D = {:.3f}".format(
                i_iter, settings.MAX_ITER,
                loss_G_seg_value, 
                loss_adv_seg_value,  
                loss_D_value))

        
        f = open(log_file, "a")
        output_log = '{:.8f},\t {:.8f},\t {:.8f}\n'.format(
            loss_G_seg_value, 
            loss_adv_seg_value,
            loss_D_value)
        f.write(output_log)
        f.close()

        # taking snapshot
        if i_iter >= settings.MAX_ITER:
            print('saving the final model ...')
            torch.save(model_G.state_dict(),osp.join(snapshot_dir, 'MODEL_'+str(settings.MAX_ITER)+'.pth'))
            torch.save(model_D.state_dict(),osp.join(snapshot_dir, 'MODEL_'+str(settings.MAX_ITER)+'_D.pth'))
            break

        if i_iter % settings.SAVE_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model_G.state_dict(),osp.join(snapshot_dir, 'MODEL_'+str(i_iter)+'.pth'))
            torch.save(model_D.state_dict(),osp.join(snapshot_dir, 'MODEL_'+str(i_iter)+'_D.pth'))
        
        
if __name__ == "__main__":
    main()








        




