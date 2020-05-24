from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import numpy as np


class AlbumentationTransforms:
    """
    Helper class to create test and train transforms using Albumentations
    """
    def __init__(self, img_transforms_list=[], bg_transforms_list=[], target_transforms_list = []):
        img_transforms_list.append(AP.ToTensor())
        bg_transforms_list.append(AP.ToTensor())
        target_transforms_list.append(AP.ToTensor())
        self.img_transforms = A.Compose(img_transforms_list)
        self.bg_transforms = A.Compose(bg_transforms_list)
        self.target_transforms = A.Compose(target_transforms_list)

    def __call__(self, sample_batch):
        img = sample_batch['image']
        bg_img = sample_batch['bg_image']
        target_img = sample_batch['target_image']
        img = np.array(img)
        bg_img = np.array(bg_img)
        target_img = np.array(target_img)
        return self.img_transforms(image=img)['image'], self.target_transforms(img = target_img)['target_image']


class Transforms:
    def __init__(self,normalize=False, mean=None, stdev=None):
        if normalize and (not mean or not stdev):
            raise ValueError('mean and stdev both are required for normalize transform')
        self.normalize = normalize
        self.mean = mean
        self.stdev = stdev

    def test_transforms(self):
        transforms_list = [transforms.ToTensor()]
        if (self.normalize):
            transforms_list.append(transforms.Normalize(self.mean, self.stdev))
        transforms_list.append(transforms.ToTensor())
        return transforms.Compose(transforms_list)

    def train_transforms(self,transforms_l=None):
        transforms_list = [transforms.ToTensor()]
        if (self.normalize):
            transforms_list.append(transforms.Normalize(self.mean, self.stdev))
        transforms_list.extend(transforms_l)
        transforms_list.append(transforms.ToTensor())
        return transforms.Compose(transforms_list)
