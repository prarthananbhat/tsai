# YoloV3
________
YoloV3 Simplified for training on Colab with custom dataset. 

_A Collage of Training images_
![image](https://github.com/theschoolofai/YoloV3/blob/master/output/train.png)


We have added a custom data set of 500 images for Hat. 

Full credit goes to [this](https://github.com/ultralytics/yolov3), and if you are looking for much more detailed explainiation and features, please refer to the original [source](https://github.com/ultralytics/yolov3). 

You'll need to download the weights from the original source. 
1. Create a folder called weights in the root (YoloV3) folder
2. Download from: https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0
3. Place 'yolov3-spp-ultralytics.pt' file in the weights folder:
4. Create your For custom dataset:
   1. Clone this repo: https://github.com/miki998/YoloV3_Annotation_Tool
   2. Follow the installation steps as mentioned in the repo. 
   3. Annotate the images using the Annotation tool. 
   4. When u annotate the images labels will be created in labels folder

5. Once your custom dataset is created create the following directory structure in your Yolo_repo

```
        data
          --customdata
            --images/ # images
              --img001.jpg
              --img002.jpg
              --...
            --labels/ labels from annotated images
              --img001.txt
              --img002.txt
              --...
            custom.data #data file
            custom.names #your class names
            custom.txt #list of name of the images you want your network to be
            trained on. Currently we are using same file for test/train
```
1. As you can see above you need to create **custom.data** file. For 1 class example, your file will look like this:

```2
  classes=1
  train=data/customdata/custom.txt
  test=data/customdata/custom.txt 
  names=data/customdata/custom.names
```
2. As you it a poor idea to keep test and train data same, but the point of this repo is to get you up and running with YoloV3 asap. You'll probably do a mistake in writing to custom.txt file. This is how our file looks like (please note the .s and /s):

```
./data/customdata/images/img001.jpg
./data/customdata/images/img002.jpg
./data/customdata/images/img003.jpg
...
```
3. You need to add custom.names file as you can see above. For our example, we downloaded images of Hat. Our custom.names file look like this:

```
Hat
```
4. Hat above will have a class index of 0. 
   6. Changes in cfg file

1. For COCO's 80 classes, VOLOv3's output vector has 255 dimensions ( (4+1+80)*3). Now we have 1 class, so we would need to change it's architecture.

2. Search for 'filters=255' (you should get entries entries). Change 255 to 18 = (4+1+1)*3

3. Search for 'classes=80' and change all three entries to 'classes=1'

4. Since you are lazy (probably), you'll be working with very few samples. In such a case it is a good idea to change:

  * burn_in to 100
  * max_batches to 5000
  * steps to 4000,4500
7. Run this command `python train.py --data data/customdata/custom.data --batch 10 --cache --cfg cfg/yolov3-custom.cfg --epochs 3 --nosave`

**Results**
After training for 300 Epochs, results look awesome!

![image](https://github.com/theschoolofai/YoloV3/blob/master/output/download.jpeg)
