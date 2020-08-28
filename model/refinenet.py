import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision

import numpy as np
import re

from model.layer_factory import conv3x3, CRPBlock, RCUBlock

class RefineNet(nn.Module):
    
    def __init__(self, pretrained=True, num_classes=38):
        super(RefineNet, self).__init__()

        self.do = nn.Dropout(p=0.5)

        # first layer before resnet
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # resent
        resnet = torchvision.models.resnet101(pretrained=pretrained)
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

        # refinenet block 2
        self.p_ims1d2_outl2_dimred = conv3x3(1024, 256, bias=False)
        self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        # refinenet block 3
        self.p_ims1d2_outl3_dimred = conv3x3(512, 256, bias=False)
        self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        # refinenet block 4
        self.p_ims1d2_outl4_dimred = conv3x3(256, 256, bias=False)
        self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)

        # get final output
        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
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

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)      
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        x1 = self.do(x1)

        out = self.clf_conv(x1)
        return out


    def optim_parameters(self, args):
        resnet_params = []
        decoder_params = []
        for k, v in self.named_parameters():
            if not v.requires_grad:
                continue
            if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
                resnet_params.append(v)
            else:
                decoder_params.append(v)
        return [{'params': resnet_params, 'lr': args.learning_rate},
                {'params': decoder_params, 'lr': 10*args.learning_rate}]


class RefineNet_middle(RefineNet):

    def __init__(self, pretrained=True, num_classes=38):
        super(RefineNet_middle, self).__init__(pretrained, num_classes)


    def forward(self, x, d):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        d = self.conv1(d)
        d = self.bn1(d)
        d = self.relu(d)
        d = self.maxpool(d)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        ld1 = self.layer1(d)
        ld2 = self.layer2(ld1)
        ld3 = self.layer3(ld2)
        ld4 = self.layer4(ld3)

        l4 = self.do(l4)
        l3 = self.do(l3)

        ld4 = self.do(ld4)
        ld3 = self.do(ld3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)
        
        d4 = self.p_ims1d2_outl1_dimred(ld4)
        d4 = self.adapt_stage1_b(d4)
        d4 = self.relu(d4)
        
        x4 = x4 + d4
                
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        
        d3 = self.p_ims1d2_outl2_dimred(ld3)
        d3 = self.adapt_stage2_b(d3)
        d3 = self.adapt_stage2_b2_joint_varout_dimred(d3)
        
        x3 = x3 + d3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        
        d2 = self.p_ims1d2_outl3_dimred(ld2)
        d2 = self.adapt_stage3_b(d2)
        d2 = self.adapt_stage3_b2_joint_varout_dimred(d2)
        
        x2 = x2 + d2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        
        d1 = self.p_ims1d2_outl4_dimred(ld1)
        d1 = self.adapt_stage4_b(d1)
        d1 = self.adapt_stage4_b2_joint_varout_dimred(d1)
        
        x1 = x1 + d1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        x1 = self.do(x1)

        out = self.clf_conv(x1)
        return out


def Segmentor(pretrained=True, num_classes=38, modality='middle'):

    model = None
    if modality == 'rgb' or modality == 'depth':
        model = RefineNet(pretrained, num_classes)
    elif modality == 'middle':
        model = RefineNet_middle(pretrained, num_classes)
    else:
        raise ValueError(modality + ' modality is not implemented')
    return model



if __name__ == "__main__":
    model = Segmentor(modality='middle')
    image = torch.randn(1, 3, 321, 321)
    depth = torch.randn(1, 3, 321, 321)
    with torch.no_grad:
        output = model(image, depth)
    print(output.size())