from PIL import Image
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data

from torchvision import transforms
import torchvision.transforms.functional as TF

import settings


class BaseDataset(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list

        self.files = []
        self.img_ids = [id.strip() for id in open(self.data_list)]
        for id in self.img_ids:
            image_file = osp.join(self.data_root, 'images/%s.png' % id)
            depth_file = osp.join(self.data_root, 'depths/%s.png' % id)
            label_file = osp.join(self.data_root, 'labels/%s.png' % id)
            self.files.append({
                "image": image_file,
                "depth": depth_file,
                "label": label_file
            })

    
    def get_data(self, idx):
        datafiles = self.files[idx]

        # read files
        image = Image.open(datafiles['image']).convert('RGB')
        depth = Image.open(datafiles['depth'])
        label = Image.open(datafiles['label']).convert('P')
        label = np.asarray(label, dtype=np.float32)

        # convert to torch tensor 
        # to_tensor gets input in range (0, 255) and returns tensor of range (0, 1) ,(not for label)
        # it also gets (H,W,C) and returns (C,H,W)
        image = TF.to_tensor(image)
        depth = TF.to_tensor(depth)
        label = TF.to_tensor(label)

        # normalize image and depth
        image = TF.normalize(image, settings.IMG_MEAN, settings.IMG_STD)
        depth = TF.normalize(depth, settings.DPT_MEAN, settings.DPT_STD)

        # add aditional dimension
        image = image.unsqueeze(dim=0)
        depth = depth.unsqueeze(dim=0)
        label = label.unsqueeze(dim=0)

        # construct a dictionary of the three outputs
        sample = {'image': image, 'depth': depth, 'label': label}

        return sample

    def __getitem__(self, idx):
        sample = self.get_data(idx)
        return sample

    def __len__(self):
        return (len(self.files))

class TrainDataset(BaseDataset):
    def __init__(self, data_root=settings.DATA_ROOT, data_list=settings.DATA_LIST):
        super(TrainDataset, self).__init__(data_root, data_list)

    def scale(self, image, depth, label):

        scale_factor = np.random.uniform(settings.SCALE_RANGE[0], settings.SCALE_RANGE[1])

        image = F.interpolate(image, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        depth = F.interpolate(depth, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        label = F.interpolate(label, scale_factor=scale_factor, mode='nearest')

        return image, depth, label

    def pad(self, image, depth, label):

        h, w = image.size()[-2:] 
        crop_size = settings.CROP_SIZE
        pad_h = max(crop_size - h, 0)
        pad_w = max(crop_size - w, 0)
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0.)
            depth = F.pad(depth, (0, pad_w, 0, pad_h), mode='constant', value=settings.DPT_IGNORE_LABEL)
            label = F.pad(label, (0, pad_w, 0, pad_h), mode='constant', value=settings.IGNORE_LABEL)

        return image, depth, label

    def crop(self, image, depth, label):


        h, w = image.size()[-2:]
        crop_size = settings.CROP_SIZE

        # get the window coordinates of crop
        s_h = np.random.randint(0, h - crop_size + 1)
        s_w = np.random.randint(0, w - crop_size + 1)
        e_h = s_h + crop_size
        e_w = s_w + crop_size

        # apply the crop
        image = image[:, :, s_h: e_h, s_w: e_w]
        depth = depth[:, :, s_h: e_h, s_w: e_w]
        label = label[:, :, s_h: e_h, s_w: e_w]

        return image, depth, label

    def flip(self, image, depth, label):

        if np.random.rand() < 0.5:
            image = torch.flip(image, [3])
            depth = torch.flip(depth, [3])
            label = torch.flip(label, [3])

        return image, depth, label



    def transform(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        image, depth, label = self.scale(image, depth, label)
        image, depth, label = self.pad(image, depth, label)
        image, depth, label = self.crop(image, depth, label)
        image, depth, label = self.flip(image, depth, label)

        return image, depth, label



    def __getitem__(self, idx):
        sample = self.get_data(idx)
        image, depth, label = self.transform(sample)

        return image[0], depth[0].repeat(3, 1, 1), label[0, 0].long()
        


class TestDataset(BaseDataset):
    def __init__(self, data_root=settings.DATA_ROOT, data_list=settings.DATA_LIST):
        super(TestDataset, self).__init__(data_root, data_list)

    def __getitem__(self, idx):
        sample = self.get_data(idx)
        image, depth, label = sample['image'], sample['depth'], sample['label']
        return image[0], depth[0].repeat(3, 1, 1), label[0, 0].long()


if __name__ == "__main__":
    traindataset = TrainDataset(data_root=settings.DATA_ROOT, data_list=settings.DATA_LIST)
    image , depth, label = traindataset.__getitem__(0)
    print("image:\t shape= {},\t min= {},\t, mean= {}, \t max= {}".format(image.shape, image.min(), image.mean(), image.max()))
    print("depth:\t shape= {},\t min= {},\t, mean= {}, \t max= {}".format(depth.shape, depth.min(), depth.mean(), depth.max()))
    print("label:\t shape= {},\t values={}".format(label.shape, torch.unique(label)))