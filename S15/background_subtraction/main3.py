from torchvision import transforms
from torchsummary import summary
import timeit
from custom_dataset import imageMaskBackgroundDataset
from data_loader import dataLoader
from utils import show_masked_data_and_bg_batch
from custom_transform import NewToTensor, NewNormalise, NewRotate, NewRandomHorizontalFlip
from accuracy import showMeasurePlots
from bg_sub_model_6channel import ResNet18
from trainmodel import trainModel
from testmodel import testModel

from config import *

# Iterating through the dataset
custom_transform = transforms.Compose([NewRotate(30),NewRandomHorizontalFlip(), NewToTensor(),
                                        NewNormalise(channel_means,channel_stdevs)
                                       ])

train_dataset  = imageMaskBackgroundDataset(csv_file = trainroot+"/train.csv",
                                  root_dir = trainroot,
                                  transform = custom_transform)
val_dataset  = imageMaskBackgroundDataset(csv_file = testroot+"/train.csv",
                                  root_dir = testroot,
                                  transform = custom_transform)

dataloader = dataLoader(device = device, batch_size=batch_size)
trainloader, testloader = dataloader(train_dataset,val_dataset)

for i_batch, sample_batched in enumerate(trainloader):
    print("Plotting images from Train Set")
    # print(sample_batched.keys())
    print("Batch:{}, Train Batch Size:{}, Test Batch Size:{}, Background Batch Size:{}".format(i_batch, sample_batched[0].size(), sample_batched[1].size(),sample_batched[2].size()))
    if i_batch == 0:
        show_masked_data_and_bg_batch(sample_batched)
        break

for i_batch, sample_batched in enumerate(testloader):
    print("Plotting images from Test Set")
    print("Batch:{}, Train Batch Size:{}, Test Batch Size:{}, Background Batch Size:{}".format(i_batch, sample_batched[0].size(),sample_batched[1].size(),sample_batched[2].size() ))
    if i_batch == 0:
        show_masked_data_and_bg_batch(sample_batched)
        break

# model = ResNet18().to(device)
# summary(model, input_size=(6, 128, 128))
#
#
# optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)
# epochs = 3
# test_losses = []
# train_losses = []
# train_batch_timer = []
# test_batch_timer = []
# train_model = trainModel()
# test_model = testModel()
# start = timeit.default_timer()
# for epoch in range(0, epochs):
#     epoch_start = timeit.default_timer()
#     print("EPOCH:",epoch)
#     train_model.train(model, device, trainloader,optimizer,train_losses,train_batch_timer)
#     test_model.test(model, device, testloader, test_losses,test_batch_timer)
#     epoch_end = timeit.default_timer()
#     print("Total Time for a epoch",(epoch_end-epoch_start))
# end = timeit.default_timer()
#
# print("Total Time:",(end - start))
#
# torch.save(model, dataroot+"model")
# showMeasurePlots.plot_loss_curves(train_losses,test_losses,epochs)
#
#
#
#
#
