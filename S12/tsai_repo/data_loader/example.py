import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from data_loader.custom_dataset import imageMaskDataset
from data_loader.utils import show_images
from data_loader.custom_transform import Rescale
from data_loader.custom_transform import ToTensor

mask_dataset = imageMaskDataset(csv_file="D:/Projects/theschoolofai/datasets/custom_dataset/data.csv",
                                root_dir = "D:/Projects/theschoolofai/datasets/custom_dataset/")

fig = plt.figure()
fig.suptitle("Sample of Raw images")
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

# Compose transforms
scale = Rescale((256,256))
composed = transforms.Compose([Rescale(256)])

fig = plt.figure()
fig.suptitle("Showing transformation on a sample")
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
        fig = plt.figure()
        fig.suptitle("Transformed Images")
        show_masked_data_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        fig.show()
        break