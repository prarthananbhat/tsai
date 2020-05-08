import os
from PIL import Image
import matplotlib.pyplot as plt
from random import randint
import datetime

from preprocess import image_preprocess
from preprocess import create_forground_mask
from preprocess import merge_fg_bg

main_path = "D:/Projects/theschoolofai/repo/tsai/S14/"
main_path = "C:/Users/pbhat/Google Drive/S14/"
bg_image_path = main_path+"background/"
fg_image_path = main_path+"foreground/"

bg_img_size = 128
fg_img_size = 32

bg_images = os.listdir(bg_image_path)
fg_images = os.listdir(fg_image_path)

start = datetime.datetime.now()
for bg_img in bg_images:
    background_image = image_preprocess(path=bg_image_path + bg_img, size=(bg_img_size, bg_img_size), channels=3)
    for fg_img in fg_images:
        # print(fg_img)
        for flip in [True,False]:
            txt = "train_{0}_{1}_{2}_".format(bg_img.split(".")[0], fg_img.split(".")[0],str(int(flip)))
            foreground_image = image_preprocess(path=fg_image_path + fg_img, size=(fg_img_size, fg_img_size), channels=4, flip=flip)
            foreground_image_mask = create_forground_mask(foreground_image)
            for pos in range(20):
                x = randint(0, (bg_img_size - fg_img_size))
                y = randint(0, (bg_img_size - fg_img_size))
                merged_image, mask_image = merge_fg_bg(background_image, foreground_image, foreground_image_mask,
                                                       position=(x, y))
                im = Image.fromarray(merged_image)
                im.save(main_path+"train/" + txt+str(pos) + ".jpg")
                plt.imsave(main_path+"mask/" + txt + str(pos) + ".jpg", mask_image, cmap='gray')
end = datetime.datetime.now()
print((end-start).seconds)