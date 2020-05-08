import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from PIL import Image
import PIL


def image_preprocess(path, size=(128, 128), channels=3, flip=False):
    if channels == 3:
        image = cv2.imread(path)
        image_RGB = image[:, :, ::-1]
    elif channels == 4:
        image = cv2.imread(path, -1)
        image_RGB = image[:, :, [2, 1, 0, 3]]
    img = cv2.resize(image_RGB, size, interpolation=cv2.INTER_AREA)
    if flip:
        img = cv2.flip(img, 1)
    return img


def create_forground_mask(image):
    img = image[:, :, 3]
    return img


def merge_fg_bg(bg_img, fg_img, fg_img_mask, position=(0, 0)):
    new_image = bg_img.copy()
    x = position[0]
    y = position[1]
    # cropping the back ground image same as the forground size
    cropped_image = new_image[x:x + 32, y:y + 32]

    # forgroung image mask  values to be converetd to 0 and1 and then reversed so that
    # keeping the forgrounds to be 0's and other regions as 1
    fg_img_mask = np.uint8(fg_img_mask / 255)
    fg_img_mask_reversed = 1 - fg_img_mask
    fg_img_mask_reversed

    # convert the mask to a 3 channel image to multiply with the 3 channel background
    img2 = cv2.merge((fg_img_mask_reversed, fg_img_mask_reversed, fg_img_mask_reversed))
    bckgrndMaskedEmptySubImg = cv2.multiply(cropped_image, img2)

    # adding the forground image to the background
    final_image = cv2.add(bckgrndMaskedEmptySubImg, fg_img[:, :, :3])

    # adding back the cropped image to the final image
    new_image[x:x + 32, y:y + 32] = final_image

    # creating the mask for final image
    bckgrndImgCopyMask = np.zeros_like(new_image[:, :, 0])
    bckgrndImgCopyMask[x:(x + 32), y:(y + 32)] = fg_img_mask

    return new_image, bckgrndImgCopyMask
