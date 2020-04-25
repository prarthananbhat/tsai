import pandas as pd
import glob

import shutil, os

##src = os.getcwd() + '/OID/dataset/train/hat'
# print(src)
##dest_dir = os.getcwd() + '/OID/dataset/train/hat_label'
# src_files = os.listdir(src)

##for filename in glob.glob(os.path.join(src, '*.txt')):
##   shutil.copy(filename, dest_dir)

# ---- For making custom.txt File -----

# src = os.getcwd() + '/OID/dataset/train/hat_images'

src = 'C:/Users/pbhat/Google Drive/YoloV3Custom/data/customdata/images/'
target_file = 'C:/Users/pbhat/Google Drive/YoloV3Custom/data/customdata/custom.txt'

src_files = os.listdir(src)

for i in range(len(src_files)):
    t = './data/customdata_detect/images/' + src_files[i] + '\n'
    print(t)

    with open(target_file, "a") as file:
        file.write(t)