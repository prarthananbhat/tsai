from __future__ import print_function
import torch
from torchvision import datasets, transforms

CUDA = torch.cuda.is_available()
print("CUDA is available:",CUDA)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if CUDA else dict(shuffle=True, batch_size=64)

# Create the data reader and transformer
def get_train_loader():
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST('./data',
                           train=True,
                           transform = train_transform,
                           download = True)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train,**dataloader_args)
    return train_loader


def get_test_loader():
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    test = datasets.MNIST('./data',
                           train=False,
                           transform = test_transform,
                           download = True)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test,**dataloader_args)
    return test_loader
