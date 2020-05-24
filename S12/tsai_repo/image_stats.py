import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import cv2

from data_loader.data_loader_tinyimagenet import dataLoader
from data_transformations.transform import AlbumentationTransforms
from utils import denormalize

#Transforms
channel_means = (0.44,0.64,0.76)
channel_stdevs = (0.36,0.19,0.21)

# Train Phase transformations
train_transforms = AlbumentationTransforms([
                                       A.Normalize(mean=channel_means, std=channel_stdevs),
                                       ])

fillmeans = (np.array(channel_means)).astype(np.uint8)

# Test Phase transformations
test_transforms = AlbumentationTransforms([A.Normalize(mean=channel_means, std=channel_stdevs)])

#Dataload
main_path = "D:/Projects/theschoolofai/datasets/custom_dataset/"
dataloader = dataLoader(path = main_path)
train_loader,classes = dataloader.get_train_loader(train_transforms,split=False, validation_split=0.3)

#Plot few images
dataiter = iter(train_loader)
images,target = next(dataiter)

print("Image shape:",images.shape)
print("Target Shape:",target.shape)

target_img_path = main_path+"target/"
ext = ".jpg"
target_img = cv2.imread(target_img_path+classes[target[0]]+ext)
print("Target Image Shape:",target_img.shape)

print("channel wise mean for single image:")
print("Mean:",images.mean())

def channel_wise_mean(images):
    min_max_list = []
    mean_list = []
    std_list = []
    for i in range(3):
        print(i)
        min_max_list.append((images[:,i,:,:].min(),images[:,i,:,:].max()))
        mean_list.append(images[:,i,:,:].mean())
        std_list.append(images[:, i, :, :].std())
    return min_max_list, mean_list, std_list


min_max_list, mean_list, std_list  = channel_wise_mean(images)
print("channel wise mean for multiple images:",mean_list)
print("channel wise range for multiple images:",min_max_list)
print("channel wise stdev for multiple images:",std_list)

"""
in this script, we calculate the image per channel mean and standard
deviation in the training set, do not calculate the statistics on the
whole dataset, as per here http://cs231n.github.io/neural-networks-2/#datapre
"""

import numpy as np
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3
def cal_dir_stat(root):
    cls_dirs = [d for d in listdir(root) if isdir(join(root, d))]
    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)
    print(cls_dirs)
    for idx, d in enumerate(cls_dirs):
        print("#{} class".format(idx))
        im_pths = glob(join(root, d, "*.jpg"))
        print(im_pths)
        for path in im_pths:
            im = cv2.imread(path)  # image in M*N*CHANNEL_NUM shape, channel in BGR order
            im = im / 255.0
            pixel_num += (im.size / CHANNEL_NUM)
            channel_sum += np.sum(im, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    return rgb_mean, rgb_std

# The script assumes that under train_root, there are separate directories for each class
# of training images.
train_root = "D:/Projects/theschoolofai/datasets/custom_dataset/new_train/"
start = timeit.default_timer()
mean, std = cal_dir_stat(train_root)
end = timeit.default_timer()
print("elapsed time: {}".format(end-start))
print("mean:{}\nstd:{}".format(mean, std))

