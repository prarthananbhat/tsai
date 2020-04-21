from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import numpy as np


class AlbumentationTransforms:
    """
    Helper class to create test and train transforms using Albumentations
    """
    def __init__(self, transforms_list=[]):
        transforms_list.append(AP.ToTensor())
        self.transforms = A.Compose(transforms_list)

    def __call__(self, img):
        img = np.array(img)
        return self.transforms(image=img)['image']


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
