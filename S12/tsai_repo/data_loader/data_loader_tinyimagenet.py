import torch
import torchvision
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
# from data_loader.custom_folder import ImageFolder


class dataLoader:
    def __init__(self,batch_size=128,num_workers=4,pin_memory=True,shuffle=True,path=None):
        # dataloader arguments - something you'll fetch these from cmdprmt
        CUDA = torch.cuda.is_available()
        print("CUDA is available:", CUDA)

        if CUDA:
            self.dataloader_args = dict(shuffle=shuffle,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               pin_memory=pin_memory)
        else:
            self.dataloader_args = dict(shuffle=True, batch_size=2)
        self.path = path
        print("Initiated Data Loader with:", self.dataloader_args)

    def get_classes(self,trainset):
        all_classes = {}
        result = {}
        for i, line in enumerate(open(self.path + 'words.txt', 'r')):
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word
        classes = []
        for key in trainset.class_to_idx:
            classes.append(all_classes[key].split(",")[0])
        return classes

    def get_train_loader(self,train_transforms=None, split=False, validation_split=0.3):
        if self.path:
            print("Loading Tinyimagenet")
            traindir = os.path.join(self.path, 'new_train')
            # traindir = self.path
            trainset = torchvision.datasets.ImageFolder(traindir, transform=train_transforms)
            # classes = self.get_classes(trainset)
            classes = trainset.classes
            if split == True:
                print("traindata is split")
                validation_split = validation_split

                test_size = int(validation_split * len(trainset))
                train_size = len(trainset) - test_size
                train_dataset, test_dataset = torch.utils.data.random_split(trainset, [train_size, test_size])
                train_loader = torch.utils.data.DataLoader(train_dataset,**self.dataloader_args)
                validation_loader = torch.utils.data.DataLoader(test_dataset,**self.dataloader_args)
                return train_loader, validation_loader#, classes
            else:
                print("Loading full trinset")
                train_loader = torch.utils.data.DataLoader(trainset, **self.dataloader_args)
                return  train_loader, classes
        else:
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                    transform=train_transforms)
            train_loader = torch.utils.data.DataLoader(trainset, **self.dataloader_args)
            return train_loader

    def get_test_loader(self,test_transforms):
        if self.path:
            testdir = os.path.join(self.path, 'test')
            testset = torchvision.datasets.ImageFolder(testdir, transform=test_transforms)
            test_loader = torch.utils.data.DataLoader(testset,**self.dataloader_args)
            return test_loader
        else:
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
            test_loader = torch.utils.data.DataLoader(testset, **self.dataloader_args)
            return test_loader

    def get_class_image(self):
        targetdir = os.path.join(self.path, 'target')
        targetset = ImageFolder(targetdir)
        return targetset

