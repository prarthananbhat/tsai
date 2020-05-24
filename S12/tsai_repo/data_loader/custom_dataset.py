import pandas as pd
from torch.utils.data import Dataset
import torch
import os
from skimage import io

class imageMaskDataset(Dataset):
    """ Image Mask DataSet"""
    def __init__(self,csv_file,root_dir,transform=None):
        """
        Args:
            csv_file (String): file with all the image paths
            root_dir (string): directory with all the images
            transform (callable, optional) optional transform to be applied to a sample
            """
        self.file_names = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,"train",self.file_names.iloc[idx,0])
        image = io.imread(img_name)
        target_img_name = os.path.join(self.root_dir, "target", self.file_names.iloc[idx, 0])
        target_image = io.imread(target_img_name)
        sample = {'image':image, 'target_image':target_image}

        if self.transform:
            sample = self.transform(sample)

        return sample
