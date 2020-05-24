import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchsummary import summary

from custom_dataset import imageMaskDataset
from custom_dataset import BackgroundDataset
from data_loader import dataLoader
from utils import show_images
from utils import show_masked_data_batch
from utils import cal_dir_stat
from custom_transform import Rescale
from custom_transform import ToTensor
from bg_sub_model import ResNet18
from custom_transform import Normalise

import torch.optim as optim
from trainmodel import train
from testmodel import test


channel_means = [0.5,0.5,0.5]
channel_stdevs = [0.2,0.2,0.2]

# mask_dataset = imageMaskDataset(csv_file="D:/Projects/theschoolofai/datasets/custom_dataset/data.csv",
#                                 root_dir = "D:/Projects/theschoolofai/datasets/custom_dataset/")
#
# fig = plt.figure()
# fig.suptitle("Sample of Raw images")
# for i in range(len(mask_dataset)):
#     sample=mask_dataset[i]
#     print(i,sample['image'].shape,sample['target_image'].shape)
#     j=i+1
#     ax = plt.subplot(3,2,j)
#     plt.tight_layout()
#     ax.set_title("Sample {}".format(i))
#     ax.axis('off')
#     show_images(**sample)
#     # print(i)
#     if i==3:
#         fig.show()
#         break
#

# Iterating through the dataset
custom_transform = transforms.Compose([
                                        ToTensor(),
                                        Normalise(channel_means,channel_stdevs)
                                       ])
dataroot = "D:/Projects/theschoolofai/datasets/background_subtraction/"
train_dataset  = imageMaskDataset(csv_file=dataroot+"traindata/output.csv",
                                  root_dir = dataroot+"traindata/",
                                  transform = custom_transform)

val_dataset  = imageMaskDataset(csv_file=dataroot+"valdata/output.csv",
                                  root_dir = dataroot+"valdata/",
                                  transform = custom_transform)

bg_transform = transforms.Compose([ToTensor(),
                                   Normalise(channel_means,channel_stdevs)])
bg_train_dataset = BackgroundDataset(csv_file=dataroot+"traindata/background.csv",
                                     root_dir=dataroot+"traindata/",
                                     transform = bg_transform)

bg_val_dataset = BackgroundDataset(csv_file=dataroot+"valdata/background.csv",
                                     root_dir=dataroot+"valdata/",
                                     transform = bg_transform)


trainloader = DataLoader(train_dataset,batch_size=8,shuffle=False)
testloader = DataLoader(val_dataset,batch_size=8,shuffle=False)
bgtrainloader = DataLoader(bg_train_dataset,batch_size=8,shuffle=False)
bgtestloader = DataLoader(bg_val_dataset,batch_size=8,shuffle=False)


for i_batch, sample_batched in enumerate(trainloader):
    print("start")
    print(i_batch, sample_batched[0].size(), sample_batched[1].size())
    # observe 4th batch and stop.
    if i_batch == 0:
        show_masked_data_batch(sample_batched,normalised=True)
        break


#Load the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ResNet18().to(device)
summary(model, input_size=(6, 128, 128))

# dataiter = iter(trainloader)
# images,target = next(dataiter)
#
# def image_stats(images):
#     images = images
#     print(type(images),images.shape)
#     print("Mean:",images.mean())
#     print("Range:",images.min(),images.max())
#     print("Standard Deviation:",images.std())
#
# image_stats(images)

optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)
epochs = 3
test_losses = []
train_losses = []
bgtrainiter, bgtestiter = iter(bgtrainloader), iter(bgtestloader)
train_bg_image, test_bg_image = next(bgtrainiter), next(bgtestiter)
train_bg_image_batch ,test_bg_image_batch= train_bg_image.repeat(8,1,1,1), test_bg_image.repeat(8,1,1,1)
print("Train Background image Shape", train_bg_image.size())
print("Train Background image Shape", train_bg_image_batch.size())
print("Test Background image Shape", test_bg_image.size())
print("Test Background image Shape", test_bg_image_batch.size())

# bg_image = bg_train_dataset[0]
plt.imshow(train_bg_image_batch[0].numpy().transpose(1,2,0))
plt.show()
plt.imshow(test_bg_image_batch[0].numpy().transpose(1,2,0))
plt.show()

for epoch in range(0, epochs):
    print("EPOCH:",epoch)
    train(model, device, trainloader,optimizer,train_losses,bg_model=train_bg_image_batch)
    test(model, device, trainloader, test_losses,bg_model=test_bg_image_batch)


# import torch.nn as nn
#
# model=model.float()
# output = model(images.float())
# target = target.unsqueeze(1)
#
# print(output.size())
# print(target.size())
#
# distance = nn.MSELoss(reduction="mean")
# loss = distance(output,target.float())
#
# print(target)
# plt_image = utils.make_grid(target)
# plt_image = plt_image.numpy().transpose(1, 2, 0)
# # plt_image = plt_image*255
# print(plt_image.min(), plt_image.max())
# plt.imshow(plt_image, cmap="gray")
# plt.show()
#
# for i_batch, sample_batched in enumerate(dataloader):
