### S15 Assignment Background subtraction
Objective: Predict forground masks

General Idea:
If we need to predict the foreground masks we need to remove the back ground from the image.
In our example the background is a sea shore and the foreground is boat.
The foreground mask will have 0's in the location where the boat is present and rest of the pixels will be black

In the previous assignment we have prepared the dataset for the same.
	1. Background images
	2. Background_foreground images
	3. Foreground masks

How does the data Look
background : 128 X 128 X3
background_with_object : 128 X 128 X 3
target_mask : 128 X 128

Basic Idea for the Model
We are expecting a 128 X 128 as output from the model and we have to input two images to the model of sixe 128 X 128 X 3reach
We know that our target is also an image which is a so we should use a pixel to pixel loss

Approach 1:

Lets us stack up both the bg and bg_fg as an input, so we have a 128 X 128 X 6 channel input
The Architecture has two parts
1. Model to extract features and detect the boat
2. Decoder to expand the features to a mask

The first model was a RESNET18 with following changes
1. Input channel changed to 6 as we stacked up the images
2. 2 convolution and maxpool layers at the start to reduce the image size from 128 to 32
3. Only 2 bloks were created and the final feature map was of 128 X 16 X 16

The seconds part was a decoder which had 3 transpose convolution layers followed by a 1 X 1 which resulted in a 128X128 as output
Loss function : MSE Loss
Total Parameters : 830,350


Further Work
The above arcihtecture was the initial model and we have to do a few changes based on the output
1. Create a background model that represents features of all the beaches which can be concatenated to the main model
2. Try out other loss functions
3. RGB Shift transformations to the images

We can do a deconvolution over the resnet output and do a pixel to pixel loss
