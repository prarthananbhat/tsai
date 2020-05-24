import os
import torch
device = "cpu"

batch_size = 8

channel_means = [0.5,0.5,0.5]
channel_stdevs = [0.2,0.2,0.2]

dataroot = "D:/Projects/theschoolofai/datasets/background_subtraction/mini_multiple_bg_multi_foreground/"
trainroot = os.path.join(dataroot,"train_data")
testroot = os.path.join(dataroot,"test_data")
