import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import utils
import torch
import numpy as np
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit

def show_images(image,target_image):
    target_image = cv2.merge((target_image, target_image, target_image))
    new_im = np.hstack((image, target_image))
    plt.imshow(new_im)

def show_images_3channel(image,target_image):
    new_im = np.hstack((image, target_image))
    plt.imshow(new_im)


def show_masked_data_batch(sample_batched,normalised=True):
    images_batch, target_images_batch = sample_batched[0], sample_batched[1]
    if normalised:
        images_batch = denormalise_batch(images_batch)
        images_batch = images_batch

    fig = plt.figure()
    fig.suptitle("Transformed Images")
    plt.subplot(2,1,1)
    grid_image = utils.make_grid(images_batch)
    x = grid_image.numpy().transpose(1, 2, 0)
    plt.imshow(x)

    plt.subplot(2, 1, 2)
    # target_images_batch = target_images_batch.numpy()
    target_images_batch = target_images_batch
    target_images_batch = np.expand_dims(target_images_batch, axis=1)
    target_images_batch = torch.from_numpy(target_images_batch)
    grid_target_image = utils.make_grid(target_images_batch)
    y = grid_target_image.numpy().transpose(1, 2, 0)
    plt.imshow(y)
    fig.show()


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


def denormalise_batch(image,channel_means = [0.5, 0.5, 0.5],channel_stdevs = [0.2, 0.2, 0.2]):
    channel_mean = np.array(channel_means)
    channel_dev = np.array(channel_stdevs)
    mean = torch.FloatTensor(channel_mean).view(1, 3, 1, 1)
    std = torch.FloatTensor(channel_dev).view(1, 3, 1, 1)
    ret = image.mul(std).add(mean)
    return ret


def image_stat(image):
    for i in range(3):
        print("channel {}".format(i))
        print("Min:{}, Max:{}, Mean:{}, STD:{}".format(image[i,:,:].min(), image[i,:,:].max(), image[i,:,:].mean(), image[i,:,:].std(ddof=1)))

def tensor_image_stat(image):
    for batch in range(4):
        for i in range(3):
            print("channel {}".format(i))
            print("Min:{}, Max:{}, Mean:{}, STD:{}".format(image[batch,i,:,:].min(), image[batch,i,:,:].max(), image[batch,i,:,:].mean(), image[batch,i,:,:].std()))


def show_image_batch(output, target):
    if output.size()[0] > 4:
        output = output[0:4, :, :, :]
        target = target[0:4, :, :, :]
    fig = plt.figure()
    fig.suptitle("Actual V/s Predicted")

    plt.subplot(2, 1, 1)
    output_img = output[0:4, :, :, :]
    # print(output_img.shape)
    output_img = output_img.detach()
    plt_image = utils.make_grid(output_img)
    plt_image = plt_image.cpu().numpy().transpose(1, 2, 0)
    print(plt_image.min(), plt_image.max())
    plt.imshow(plt_image, cmap="gray")

    plt.subplot(2, 1, 2)
    # print(target.shape)
    target_img = target[0:4, :, :, :]
    # print(target_img.shape)
    plt_image = utils.make_grid(target_img)
    plt_image = plt_image.cpu().numpy().transpose(1, 2, 0)
    # plt_image = plt_image*255
    print(plt_image.min(), plt_image.max())
    plt.imshow(plt_image, cmap="gray")
    fig.show()


def show_background_image(train_bg_image_batch,test_bg_image_batch, normalised=True):
    print("Plotting Background images")
    if normalised:
        train_bg_image_batch = denormalise_batch(train_bg_image_batch)
        test_bg_image_batch = denormalise_batch(test_bg_image_batch)
    plt.imshow(train_bg_image_batch[0].numpy().transpose(1,2,0))
    plt.show()
    plt.imshow(test_bg_image_batch[0].numpy().transpose(1,2,0))
    plt.show()

def process_bg_image(bgtrainloader,bgtestloader,batchsize = 8 ):
    print("Processing Background images")
    bgtrainiter, bgtestiter = iter(bgtrainloader), iter(bgtestloader)
    train_bg_image, test_bg_image = next(bgtrainiter), next(bgtestiter)

    train_bg_image_batch ,test_bg_image_batch= train_bg_image.repeat(batchsize,1,1,1), test_bg_image.repeat(batchsize,1,1,1)
    # print("Train Background image Shape", train_bg_image.size())
    print("Train Background image Shape", train_bg_image_batch.size())
    # print("Test Background image Shape", test_bg_image.size())
    print("Test Background image Shape", test_bg_image_batch.size())
    return train_bg_image_batch, test_bg_image_batch


def show_masked_data_and_bg_batch(sample_batched,normalised=True):
    images_batch, target_images_batch, bg_image_batch = sample_batched[0], sample_batched[1], sample_batched[2]
    if normalised:
        images_batch = denormalise_batch(images_batch)

    fig = plt.figure()
    fig.suptitle("Transformed Images")
    plt.subplot(3,1,1)
    grid_image = utils.make_grid(images_batch)
    x = grid_image.numpy().transpose(1, 2, 0)
    plt.imshow(x)

    plt.subplot(3, 1, 2)
    # target_images_batch = target_images_batch.numpy()
    target_images_batch = target_images_batch
    target_images_batch = np.expand_dims(target_images_batch, axis=1)
    target_images_batch = torch.from_numpy(target_images_batch)
    grid_target_image = utils.make_grid(target_images_batch)
    y = grid_target_image.numpy().transpose(1, 2, 0)
    plt.imshow(y)

    if normalised:
        bg_image_batch = denormalise_batch(bg_image_batch)
    plt.subplot(3,1,3)
    grid_image = utils.make_grid(bg_image_batch)
    z = grid_image.numpy().transpose(1, 2, 0)
    plt.imshow(z)
    fig.show()
