import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import torch
import torch.optim as optim
import cv2

from data_loader.data_loader_tinyimagenet import dataLoader
from data_transformations.transform import AlbumentationTransforms
from models.bg_sub_model import ResNet18
from utils import denormalize
from training.training import custom_train
from scoring.scoring import test
from scoring.accuracy import showMeasurePlots
from scoring.missclassified_images import identifyImages

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
# dataloader = dataLoader(path = "D:/Projects/theschoolofai/datasets/custom_dataset/target")
# target_loader = dataloader.get_train_loader(train_transforms,split=False, validation_split=0.3)

#Plot few images
dataiter = iter(train_loader)
images,target = next(dataiter)

print(images.shape)

print(target.shape)
print(target)
print(classes[target[0]])
print(classes[target[1]])

target_img_path = main_path+"target/"
ext = ".jpg"
print(target_img_path+classes[target[0]])
target_img = cv2.imread(target_img_path+classes[target[0]]+ext)

# If the picture in sormalised then denormalise and plot it
def show_image(img,title=None):
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])

fig = plt.figure()
for i in range(0,2):
    x = images[i]
    plt_image = denormalize(x,channel_means,channel_stdevs)
    plt_image = np.transpose(plt_image,[1,2,0])
    plt.subplot(2,2,2*i+1)
    show_image(plt_image,title = classes[target[i]])
    target_img = cv2.imread(target_img_path+classes[target[i]]+ext)
    target_img=np.uint8(target_img)
    plt.subplot(2,2,2*i+2)
    show_image(target_img)
fig.show()


#Load the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ResNet18().to(device)
# summary(model, input_size=(3, 128, 128))

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
epochs = 10
test_losses = []
train_losses = []
test_accuracy = []
train_accuracy = []
for epoch in range(0, epochs):
    print("EPOCH:",epoch)
    custom_train(model, device, train_loader, classes,optimizer,train_losses,train_accuracy)
    test(model, device, train_loader,test_losses,test_accuracy)

showMeasurePlots.plot_accuracy_curves(train_accuracy,test_accuracy,epochs)
showMeasurePlots.plot_loss_curves(train_accuracy,test_accuracy,epochs)
