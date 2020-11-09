import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision

import numpy as np
import re

import settings

from model.layer_factory import conv3x3, CRPBlock, RCUBlock

class RefineNet(nn.Module):
    
    def __init__(self, pretrained=True, num_classes=38):
        super(RefineNet, self).__init__()

        self.do = nn.Dropout(p=0.5)
        self.dpt_do = nn.Dropout(p=0.5)

        # first layer before resnet
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dpt_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.dpt_bn1 = nn.BatchNorm2d(64)
        self.dpt_relu = nn.ReLU(inplace=True)
        self.dpt_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # encoder (resent)
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


        # refinenet block 1
        self.p_ims1d2_outl1_dimred = conv3x3(2048, 512, bias=False)
        self.adapt_stage1_b = self._make_rcu(512, 512, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3(512, 256, bias=False)

        self.dpt_p_ims1d2_outl1_dimred = conv3x3(2048, 512, bias=False)
        self.dpt_adapt_stage1_b = self._make_rcu(512, 512, 2, 2)
        self.dpt_mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.dpt_mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.dpt_mflow_conv_g1_b3_joint_varout_dimred = conv3x3(512, 256, bias=False)


        # refinenet block 2
        self.p_ims1d2_outl2_dimred = conv3x3(1024, 256, bias=False)
        self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.dpt_p_ims1d2_outl2_dimred = conv3x3(1024, 256, bias=False)
        self.dpt_adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
        self.dpt_adapt_stage2_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.dpt_mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.dpt_mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.dpt_mflow_conv_g2_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)


        # refinenet block 3
        self.p_ims1d2_outl3_dimred = conv3x3(512, 256, bias=False)
        self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.dpt_p_ims1d2_outl3_dimred = conv3x3(512, 256, bias=False)
        self.dpt_adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
        self.dpt_adapt_stage3_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.dpt_mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.dpt_mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.dpt_mflow_conv_g3_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        # refinenet block 4
        self.p_ims1d2_outl4_dimred = conv3x3(256, 256, bias=False)
        self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)

        self.dpt_p_ims1d2_outl4_dimred = conv3x3(256, 256, bias=False)
        self.dpt_adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
        self.dpt_adapt_stage4_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.dpt_mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.dpt_mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)

        # get final output
        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)

        self.dpt_conv = nn.Conv2d(256, 1, kernel_size=3, stride=1,
                                    padding=1, bias=True)


    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.Sequential(*layers)
        

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)

        # stream 1
        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)      
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)

        d4 = self.dpt_p_ims1d2_outl1_dimred(l4)
        d4 = self.dpt_adapt_stage1_b(d4)
        d4 = self.dpt_relu(d4)
        d4 = self.dpt_mflow_conv_g1_pool(d4)
        d4 = self.dpt_mflow_conv_g1_b(d4)
        d4 = self.dpt_mflow_conv_g1_b3_joint_varout_dimred(d4)
        d4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(d4)


        # stream 2
        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)

        d3 = self.dpt_p_ims1d2_outl2_dimred(l3)
        d3 = self.dpt_adapt_stage2_b(d3)
        d3 = self.dpt_adapt_stage2_b2_joint_varout_dimred(d3)
        d3 = d3 + d4
        d3 = F.relu(d3)
        d3 = self.dpt_mflow_conv_g2_pool(d3)
        d3 = self.dpt_mflow_conv_g2_b(d3)
        d3 = self.dpt_mflow_conv_g2_b3_joint_varout_dimred(d3)
        d3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(d3)

        
        # stream 3
        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        d2 = self.dpt_p_ims1d2_outl3_dimred(l2)
        d2 = self.dpt_adapt_stage3_b(d2)
        d2 = self.dpt_adapt_stage3_b2_joint_varout_dimred(d2)
        d2 = d2 + d3
        d2 = F.relu(d2)
        d2 = self.dpt_mflow_conv_g3_pool(d2)
        d2 = self.dpt_mflow_conv_g3_b(d2)
        d2 = self.dpt_mflow_conv_g3_b3_joint_varout_dimred(d2)
        d2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(d2)

        # stream 4
        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        x1 = self.do(x1)

        d1 = self.dpt_p_ims1d2_outl4_dimred(l1)
        d1 = self.dpt_adapt_stage4_b(d1)
        d1 = self.dpt_adapt_stage4_b2_joint_varout_dimred(d1)
        d1 = d1 + d2
        d1 = F.relu(d1)
        d1 = self.dpt_mflow_conv_g4_pool(d1)
        d1 = self.dpt_mflow_conv_g4_b(d1)
        d1 = self.dpt_do(d1)

        out_seg = self.clf_conv(x1)
        out_dpt = self.dpt_conv(d1)
        return out_seg, out_dpt


    def optim_parameters(self, lr):
        resnet_params = []
        decoder_params = []
        for k, v in self.named_parameters():
            if not v.requires_grad:
                continue
            if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
                resnet_params.append(v)
            else:
                decoder_params.append(v)
        return [{'params': resnet_params, 'lr': lr},
                {'params': decoder_params, 'lr': 10*lr}]
    
    # consider seg and dpt learning rates separately
    # def optim_parameters(self, args):
    #     resnet_params = []
    #     decoder_params = []
    #     dpt_decoder_params = []
        
    #     for k, v in self.named_parameters():
    #         if not v.requires_grad:
    #             continue
    #         if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
    #             resnet_params.append(v)
    #         else:
    #             if bool(re.match(".*dpt_.*", k)):
    #                 dpt_decoder_params.append(v)
    #             else:
    #                 decoder_params.append(v)
    #     return [{'params': resnet_params, 'lr': args.learning_rate},
    #             {'params': decoder_params, 'lr': 10*args.learning_rate},
    #             # {'params': dpt_resnet_params, 'lr': args.learning_rate_dpt},
    #             {'params': dpt_decoder_params, 'lr': 10*args.learning_rate_dpt}]



def Segmentor(pretrained=True, num_classes=38, modality='rgb'):

    # model = None
    # if modality == 'rgb' or modality == 'depth':
    #     model = RefineNet(pretrained, num_classes)
    # elif modality == 'middle':
    #     model = RefineNet(pretrained, num_classes)
    # else:
    #     raise ValueError(modality + ' modality is not implemented')
    model = RefineNet(pretrained, num_classes)
    return model



if __name__ == "__main__":
    model = Segmentor(modality='middle')
    image = torch.randn(1, 3, 350, 350)
    depth = torch.randn(1, 3, 321, 321)
    with torch.no_grad:
        out_seg, out_dpt = model(image)
    print(out_seg.size(), out_dpt.size())