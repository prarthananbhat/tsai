from skimage import transform
import torch
import numpy as np
from utils import image_stat
from torchvision.transforms import functional as F
import random
import numbers



class Rescale(object):
    """Resclae the image to the desired shape
    Args:
        output size (tuple or int): Desired output size, If tuple output is matched to output size
        if int then smaller of image edges is matched to outputsize keeping aspect ratio the same
        """
    def __init__(self,output_size):
        # assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, target_image = sample['image'],sample['target_image']
        h,w = image.shape[:2]
        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size * h / w,self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h , new_w = int(new_h), int(new_w)
        img = transform.resize(image,(new_h, new_w))
        target_img = transform.resize(target_image,(new_h, new_w))

        return {'image':img, 'target_image': target_img}

# class ToTensor(object):
#     """ converts ndarrays in samples to tensors"""
#     def __call__(self, sample):
#         image , target_image = sample['image'], sample['target_image']
#         image = image.transpose((2,0,1))
#         image = image/255
#         target_image = target_image/255.0
#         # target_image = target_image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'target_image': torch.from_numpy(target_image)}


# class Normalise(object):
#     def __init__(self,image_channel_mean,image_channel_std):
#         self.image_channel_mean = torch.tensor(image_channel_mean)
#         self.image_channel_std = torch.tensor(image_channel_std)
#
#     def __call__(self, sample):
#         image_tensor = sample['image']
#         target_tensor = sample['target_image']
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#         Returns:
#             Tensor: Normalized image.
#         """
#         for t, m, s in zip(image_tensor, self.image_channel_mean, self.image_channel_std):
#             # t.mul_(s).add_(m)
#             t.sub_(m).div_(s)
#         return image_tensor,target_tensor

class ToTensor(object):
    """ converts ndarrays in samples to tensors"""
    def __call__(self, sample):
        image  = sample['image']
        image = image.transpose((2,0,1))
        image = image/255
        if 'target_image' in sample:
            target_image = sample['target_image']
            target_image = target_image / 255.0
            return {'image': torch.from_numpy(image),
                    'target_image': torch.from_numpy(target_image)}
        else:
            return {'image': torch.from_numpy(image)}

class Normalise(object):
    def __init__(self,image_channel_mean,image_channel_std):
        self.image_channel_mean = torch.tensor(image_channel_mean)
        self.image_channel_std = torch.tensor(image_channel_std)

    def __call__(self, sample):
        image_tensor = sample['image']
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(image_tensor, self.image_channel_mean, self.image_channel_std):
            # t.mul_(s).add_(m)
            t.sub_(m).div_(s)
        if 'target_image' in sample:
            target_tensor = sample['target_image']
            return image_tensor,target_tensor
        else:
            return image_tensor


class NewNormalise(object):
    def __init__(self,image_channel_mean,image_channel_std):
        self.image_channel_mean = torch.tensor(image_channel_mean)
        self.image_channel_std = torch.tensor(image_channel_std)

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if 'image' in sample:
            image_tensor = sample['image']
            image_tensor = self.normalise(image_tensor)
        if 'target_image' in sample:
            target_tensor = sample['target_image']
        if 'bg_image' in sample:
            bg_tensor = sample['bg_image']
            bg_tensor = self.normalise(bg_tensor)
        if 'target_image' in sample:
            target_tensor = sample['target_image']

        if target_tensor is None and bg_tensor is None:
            return image_tensor
        if bg_tensor is None:
            return image_tensor, target_tensor
        else:
            return image_tensor,target_tensor,bg_tensor

    def normalise(self, tensor):
        for t, m, s in zip(tensor, self.image_channel_mean, self.image_channel_std):
            # t.mul_(s).add_(m)
            t.sub_(m).div_(s)
        return tensor

class NewToTensor(object):
    """ converts ndarrays in samples to tensors"""
    def __call__(self, sample):
        if 'image' in sample:
            image  = np.array(sample['image'])
            image = image.transpose((2,0,1))
            image = image/255
        if 'target_image' in sample:
            target_image = np.array(sample['target_image'])
            target_image = target_image / 255.0
        if 'bg_image' in sample:
            bg_image = np.array(sample['bg_image'])
            bg_image = bg_image.transpose((2,0,1))
            bg_image = bg_image/255

        if target_image is None and bg_image is None:
            return {'image': torch.from_numpy(image)}

        if bg_image is None:
            return {'image': torch.from_numpy(image),
                    'target_image': torch.from_numpy(target_image)}
        else:
            return {'image': torch.from_numpy(image),
                    'target_image': torch.from_numpy(target_image),
                    'bg_image': torch.from_numpy(bg_image)}

class NewRotate(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center


    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, sample):
        angle = self.get_params(self.degrees)

        if 'image' in sample:
            image  = sample['image']
            ret_image = F.rotate(image, angle, self.resample, self.expand, self.center)
        if 'target_image' in sample:
            target_image = sample['target_image']
            ret_target_image = F.rotate(target_image, angle, self.resample, self.expand, self.center)
        if 'bg_image' in sample:
            bg_image = sample['bg_image']
            # ret_bg_image = F.rotate(bg_image, angle, self.resample, self.expand, self.center)

        return {'image': ret_image,
                    'target_image': ret_target_image,
                    'bg_image': bg_image}

class NewRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        img = sample['image']
        target_image = sample['target_image']
        bg_image = sample['bg_image']

        if random.random() < self.p:
            ret_bg_img = F.hflip(bg_image)
            return {'image': img,
                    'target_image': target_image,
                    'bg_image': ret_bg_img}
        return {'image': img,
                'target_image': target_image,
                'bg_image': bg_image}

