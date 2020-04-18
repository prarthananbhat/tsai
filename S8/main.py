from __future__ import print_function

import torch
from torchsummary import summary
import torch.optim as optim

from models.resnet18 import ResNet18
from data_loader.data_loader_cifar import get_train_loader
from data_loader.data_loader_cifar import get_test_loader
from training.training import train
from scoring.scoring import test


# Set seed for all the environments
SEED = 1
torch.manual_seed(SEED)

CUDA = torch.cuda.is_available()
print("CUDA is available:",CUDA)
# If CUDA is available the set SEED for it
if CUDA:
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if CUDA else "cpu")
print(device)
model = ResNet18().to(device)
summary(model, input_size=(3, 32, 32))

from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []




train_loader = get_train_loader()
test_loader = get_test_loader()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
EPOCHS = 2
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer, epoch,train_losses,train_acc)
    test(model, device, test_loader,test_losses,test_acc)