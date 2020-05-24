from torchvision import transforms, utils
from torchsummary import summary

from custom_dataset import imageMaskDataset
from custom_dataset import BackgroundDataset
from data_loader import dataLoader
from utils import show_masked_data_batch
from utils import show_background_image
from utils import process_bg_image
from custom_transform import ToTensor
from bg_sub_model_6channel import ResNet18
from custom_transform import Normalise
from accuracy import showMeasurePlots
from trainmodel import trainModel
from testmodel import testModel
from config import *

# Iterating through the dataset
custom_transform = transforms.Compose([ToTensor(),
                                        Normalise(channel_means,channel_stdevs)
                                       ])

train_dataset  = imageMaskDataset(csv_file = trainroot+"/train.csv",
                                  root_dir = trainroot,
                                  transform = custom_transform)

val_dataset  = imageMaskDataset(csv_file = testroot+"/train.csv",
                                  root_dir = testroot,
                                  transform = custom_transform)

bg_transform = transforms.Compose([ToTensor(),
                                   Normalise(channel_means,channel_stdevs)])
bg_train_dataset = BackgroundDataset(csv_file= trainroot+"/background.csv",
                                     root_dir= trainroot,
                                     transform = bg_transform)

bg_val_dataset = BackgroundDataset(csv_file=testroot+"/background.csv",
                                     root_dir=testroot,
                                     transform = bg_transform)

dataloader = dataLoader(device = device, batch_size=batch_size)
trainloader, testloader = dataloader(train_dataset,val_dataset)
bgtrainloader, bgtestloader = dataloader(bg_train_dataset,bg_val_dataset)



for i_batch, sample_batched in enumerate(trainloader):
    print("Plotting images from Train Set")
    print("Batch:{}, Train Batch Size:{}, Test Batch Size:{}".format(i_batch, sample_batched[0].size(), sample_batched[1].size()))
    if i_batch == 0:
        show_masked_data_batch(sample_batched)
        break

for i_batch, sample_batched in enumerate(testloader):
    print("Plotting images from Test Set")
    print("Batch:{}, Train Batch Size:{}, Test Batch Size:{}".format(i_batch, sample_batched[0].size(), sample_batched[1].size()))
    if i_batch == 0:
        show_masked_data_batch(sample_batched)
        break

train_bg_image_batch, test_bg_image_batch = process_bg_image(bgtrainloader,bgtestloader,batchsize=batch_size)
show_background_image(train_bg_image_batch,test_bg_image_batch)

#Load the model
model = ResNet18().to(device)
summary(model, input_size=(6, 128, 128))

optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)
epochs = 3
test_losses = []
train_losses = []
train_model = trainModel()
test_model = testModel()
for epoch in range(0, epochs):
    print("EPOCH:",epoch)
    train_model.train(model, device, trainloader,optimizer,train_losses,bg_model=train_bg_image_batch)
    test_model.test(model, device, trainloader, test_losses,bg_model=test_bg_image_batch)

showMeasurePlots.plot_loss_curves(train_losses,test_losses,epochs)


