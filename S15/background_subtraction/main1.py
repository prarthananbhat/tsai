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
from testmodel import testModel


channel_means = [0.5,0.5,0.5]
channel_stdevs = [0.2,0.2,0.2]
dataroot = "D:/Projects/theschoolofai/datasets/background_subtraction/"

# Iterating through the dataset
custom_transform = transforms.Compose([ToTensor(),
                                       Normalise(channel_means,channel_stdevs)
                                       ])

train_dataset  = imageMaskDataset(csv_file=dataroot+"fulltraindata/output.csv",
                                  root_dir = dataroot+"fulltraindata/",
                                  transform = custom_transform)

trainloader = DataLoader(train_dataset,batch_size=10,shuffle=False)


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
summary(model, input_size=(3, 128, 128))


optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)
epochs = 3
test_losses = []
train_losses = []


for epoch in range(0, epochs):
    print("EPOCH:",epoch)
    train(model, device, trainloader,optimizer,train_losses)
    testmodel = testModel()
    testmodel.test(model,device,trainloader,test_losses)


