import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import torch
import torch.optim as optim

from data_loader.data_loader_cifar import dataLoader
from data_transformations.transform import AlbumentationTransforms
from models.resnet18 import ResNet18
from utils import denormalize
from training.training import train
from scoring.scoring import test
from scoring.accuracy import showMeasurePlots
from scoring.missclassified_images import identifyImages

#Transforms
channel_means = (0.49139968, 0.48215841, 0.44653091)
channel_stdevs = (0.24703223, 0.24348513, 0.26158784)
# Train Phase transformations
train_transforms = AlbumentationTransforms([
                                       A.Rotate((-30.0, 30.0)),
                                       A.HorizontalFlip(),
                                       A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
                                       A.Normalize(mean=channel_means, std=channel_stdevs),
                                       A.Cutout(num_holes=4) # fillvalue is 0 after normalizing as mean is 0
                                       ])

fillmeans = (np.array(channel_means)).astype(np.uint8)

# Test Phase transformations
test_transforms = AlbumentationTransforms([A.Normalize(mean=channel_means, std=channel_stdevs)])

#Dataload
dataloader = dataLoader()
train_loader = dataloader.get_train_loader(train_transforms)
test_loader = dataloader.get_train_loader(test_transforms)
classes = dataloader.get_classes()
print(train_loader.dataset)
print(classes)

# #Plot few images
# dataiter = iter(train_loader)
# images, target = next(dataiter)
# print(images[0].shape)
#
# #If the picture in sormalised then denormalise and plot it
# fig = plt.figure()
# for i in range(0,6):
#     plt_image = denormalize(images[i],channel_means,channel_stdevs)
#     plt_image = np.transpose(plt_image,[2,1,0])
#     plt.subplot(2,3,i+1)
#     plt.imshow(plt_image)
#     plt.title(str(classes[target[i]]))
#     plt.xticks([])
#     plt.yticks([])
# fig.show()
#
# #Load the model
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# model = ResNet18().to(device)
# summary(model, input_size=(3, 32, 32))
#
# epochs = 1
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# test_losses = []
# train_losses = []
# test_accuracy = []
# train_accuracy = []
# for epoch in range(0, epochs):
#     print("EPOCH:",epoch)
#     train(model, device, train_loader, optimizer, epoch,train_losses,train_accuracy,L1lambda=0.001)
#     test(model, device, test_loader,test_losses,test_accuracy)
#
# showMeasurePlots.plot_accuracy_curves(train_accuracy,test_accuracy,epochs)
# showMeasurePlots.plot_loss_curves(train_accuracy,test_accuracy,epochs)
#
# misclass_data_r, misclass_targets_r,misclass_pred_r = identifyImages.misclassified(model,test_loader,device)
# correctclass_data_r, correctclass_targets_r,correctclass_pred_r = identifyImages.correct_classified(model,test_loader,device)
#
# from matplotlib.pyplot import figure
# fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
# misclass_targets_r_cpu = misclass_targets_r.cpu().numpy()
# misclass_pred_r_cpu = misclass_pred_r.cpu().numpy()
# for num in range(0,20):
#     plt.subplot(5,5,num+1)
#     plt.tight_layout()
#     mis_class_img = misclass_data_r[num]
#     mis_class_img_cpu = mis_class_img.cpu()
#     plt_image = np.transpose(mis_class_img_cpu, (1, 2, 0))
#     plt.imshow(plt_image)
#     plt.xticks([])
#     plt.yticks([])
#     plt.title("Predicted: {} \n Actual:{}".format(
#     classes[misclass_pred_r_cpu[num]], classes[misclass_targets_r_cpu[num]]))
#
# from matplotlib.pyplot import figure
# fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
# correctclass_targets_r_cpu = correctclass_targets_r.cpu().numpy()
# correctclass_pred_r_cpu = correctclass_pred_r.cpu().numpy()
# for num in range(0,20):
#     plt.subplot(5,5,num+1)
#     plt.tight_layout()
#     correct_class_img = correctclass_data_r[num]
#     correct_class_img_cpu = correct_class_img.cpu()
#     plt_image = np.transpose(correct_class_img_cpu, (1, 2, 0))
#     plt.imshow(plt_image)
#     plt.xticks([])
#     plt.yticks([])
#     plt.title("Predicted: {} \n Actual:{}".format(
#     classes[correctclass_pred_r_cpu[num]], classes[correctclass_targets_r_cpu[num]]))
#
#
