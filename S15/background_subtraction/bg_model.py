import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import os

# path_to_background_images
dataroot = "D:/Projects/theschoolofai/datasets/background_subtraction/mini_single_bg_multi_foreground/"
bgpath = os.path.join(dataroot,"resize_background")

bg_transform = transforms.Compose([ToTensor(),
                                   Normalise(channel_means,channel_stdevs)])
bg_train_dataset = BackgroundDataset(csv_file= trainroot+"/background.csv",
                                     root_dir= trainroot,
                                     transform = bg_transform)
