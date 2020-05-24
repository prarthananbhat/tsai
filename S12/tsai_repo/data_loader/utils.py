import numpy as np
import matplotlib.pyplot as plt

def show_images(image,target_image):
    new_im = np.hstack((image, target_image))
    plt.imshow(new_im)
