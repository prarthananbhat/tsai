import torch
import torchvision


class dataLoader:
    def __init__(self,batch_size=128,num_workers=4,pin_memory=True,shuffle=True):
        # dataloader arguments - something you'll fetch these from cmdprmt
        CUDA = torch.cuda.is_available()
        print("CUDA is available:", CUDA)

        if CUDA:
            self.dataloader_args = dict(shuffle=shuffle,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               pin_memory=pin_memory)
        else:
            self.dataloader_args = dict(shuffle=True, batch_size=64)
        print("Initiated Data Loader with:", self.dataloader_args)

    def get_train_loader(self,train_transforms):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=train_transforms)
        train_loader = torch.utils.data.DataLoader(trainset, **self.dataloader_args)
        return train_loader

    def get_test_loader(self,test_transforms):
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=test_transforms)
        test_loader = torch.utils.data.DataLoader(testset,**self.dataloader_args)
        return test_loader

    def get_classes(self):
        classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return classes

