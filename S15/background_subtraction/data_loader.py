import torch
import timeit

class dataLoader:
    def __init__(self,device, batch_size=128,num_workers=4,pin_memory=True,shuffle=True):
        # dataloader arguments - something you'll fetch these from cmdprmt
        start = timeit.default_timer()
        if device == "cuda":
            self.dataloader_args = dict(shuffle=shuffle,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               pin_memory=pin_memory)
        else:
            self.dataloader_args = dict(shuffle=True, batch_size=8)
        initiate_time = timeit.default_timer()-start
        print("Initiated Data Loader with:", self.dataloader_args)
        print("Time Taken:", initiate_time)

    def __call__(self, train_dataset, test_dataset):
        start = timeit.default_timer()
        train_loader = torch.utils.data.DataLoader(train_dataset,**self.dataloader_args)
        test_loader = torch.utils.data.DataLoader(test_dataset,**self.dataloader_args)
        print("Data Load Time:",timeit.default_timer()-start)
        return train_loader, test_loader
