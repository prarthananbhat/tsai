import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import os
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, utils

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


def show_images(image,target_image):
    new_im = np.hstack((image, target_image))
    plt.imshow(new_im)


mask_dataset = imageMaskDataset(csv_file="D:/Projects/theschoolofai/datasets/custom_dataset/data.csv",
                                root_dir = "D:/Projects/theschoolofai/datasets/custom_dataset/")
plt.close('all')
fig = plt.figure()
for i in range(len(mask_dataset)):
    sample=mask_dataset[i]
    print(i,sample['image'].shape,sample['target_image'].shape)
    j=i+1
    ax = plt.subplot(3,2,j)
    plt.tight_layout()
    ax.set_title("Sample {}".format(i))
    ax.axis('off')
    show_images(**sample)
    # print(i)
    if i==3:
        fig.show()
        break



class Rescale(object):
    """Resclae the image to the desired shape
    Args:
        output size (tuple or int): Desired output size, If tuple output is matched to output size
        if int then smaller of image edges is matched to outputsize keeping aspect ratio the same
        """
    def __init__(self,output_size):
        # assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, target_image = sample['image'],sample['target_image']
        h,w = image.shape[:2]
        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size * h / w,self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h , new_w = int(new_h), int(new_w)
        img = transform.resize(image,(new_h, new_w))
        target_img = transform.resize(target_image,(new_h, new_w))

        return {'image':img, 'target_image': target_img}

class ToTensor(object):
    """ converts ndarrays in samples to tensors"""
    def __call__(self, sample):
        image , target_image = sample['image'], sample['target_image']
        image = image.transpose((2,0,1))
        target_image = target_image.transpose((2, 0, 1))
        return image, target_image

# Compose transforms
scale = Rescale((256,256))
composed = transforms.Compose([Rescale(256)])

fig = plt.figure()
sample = mask_dataset[0]
print(sample.keys())
for i, trsfm in enumerate([scale,composed]):
    trsansformed_sample = trsfm(sample)
    ax = plt.subplot(1,2,i+1)
    plt.tight_layout()
    ax.set_title(type(trsfm).__name__)
    show_images(**trsansformed_sample)
fig.show()


#Iterating through the dataset
custom_transform = transforms.Compose([Rescale(256),ToTensor()])
transformed_dataset = mask_dataset = imageMaskDataset(csv_file="D:/Projects/theschoolofai/datasets/custom_dataset/data.csv",
                                root_dir = "D:/Projects/theschoolofai/datasets/custom_dataset/",
                                                      transform = custom_transform)

dataloader = DataLoader(transformed_dataset,batch_size=4,shuffle=True)

#Helper function to show a batch
def show_masked_data_batch(sample_batched):
    images_batch, target_images_batch = sample_batched[0], sample_batched[1]
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid_image = utils.make_grid(images_batch)
    # plt.imshow(grid.numpy().transpose(1,2,0))
    grid_target_image = utils.make_grid(target_images_batch)
    x = grid_image.numpy().transpose(1, 2, 0)
    y = grid_target_image.numpy().transpose(1, 2, 0)
    new_im = np.vstack((x, y))
    plt.imshow(new_im)


for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched[0].size(), sample_batched[1].size())
    # observe 4th batch and stop.
    if i_batch == 0:
        plt.figure()
        show_masked_data_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break

