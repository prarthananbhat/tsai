### Objective: Preparing Custom Data set for Background Subtraction and Depth Perdiction

*Data set for background subtraction shoulds have traning images of a object and a background where as target will have only the object
Data set for depth prediction should have traning images of a object and a background where as target will have the depth map for the same.*

#### Steps to create the data set
1. Download the back ground images of your choice and save the images in the sample background folder, In this example we have chosen a background of beach.  

![alt text](https://github.com/prarthananbhat/tsai/blob/master/S14/images/background.png)

2. Choose a foreground object of your choice and save this image in sample foreground folder, selected boats in this example.
3. We have to remove the back ground from this image so that we can place the boat over the background image. We have used Microsoft Powerpoint to remove the background, or any tools similar to photoshop can be used to do the same.

![alt text](https://github.com/prarthananbhat/tsai/blob/master/S14/images/foreground_transperant.png)

4. Save this image as png so that we have the 4th channel.
5. Resisie the forground image to the desired shape and extract only the 4th channel which gives you the mask

![alt text](https://github.com/prarthananbhat/tsai/blob/master/S14/images/foreground_mask.png)

7. Resise the back ground to a desired shape and choose a random position on your background image of size same as the foreground image. We have a background of size 128X128 and foreground as 32X32.
8. The chosen area of the background is replaced with with the foreground image using the mask.

![alt text](https://github.com/prarthananbhat/tsai/blob/master/S14/images/final_image.png)

9. Using the mask create the mask for the entire image.

![alt text](https://github.com/prarthananbhat/tsai/blob/master/S14/images/final_mask.png)

Refer the notebook to create a sample data [Notebook](https://github.com/prarthananbhat/tsai/blob/master/S14/dataset_creation.ipynb)

In this repo we have some sample images to try on.

1. [Sample Background](https://github.com/prarthananbhat/tsai/tree/master/S14/sample_background)

2. [Sample foreground](https://github.com/prarthananbhat/tsai/tree/master/S14/sample_foreground)

The following two folders have the final images and the mask

1. [Sample final images](https://github.com/prarthananbhat/tsai/tree/master/S14/sample_train)

2. [Sample masks](https://github.com/prarthananbhat/tsai/tree/master/S14/sample_mask)

Refer this notebook for creating the depth maps [Notebook](https://github.com/prarthananbhat/tsai/blob/master/S14/depth_images.ipynb)
Link to the complete data is here [Complete Data Set](https://drive.google.com/drive/folders/1_I4TMyLBlKtupTWRpYJMge2zQtdsJ17o?usp=sharing)

