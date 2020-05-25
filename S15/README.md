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

#### Approach 1: RESNEt18 with alterations
1. Data Transformations applied:
	1. Random Rotation
	2. Horizontal Flip

2. Stack up both the bg and bg_fg as an input, so we have a 128 X 128 X 6 channel input
3. The Architecture has two parts
	1. Feature Extractor : The first model was a RESNET18 with following changes
		1. Input channel changed to 6 as we stacked up the images
		2. 2 convolution and maxpool layers at the start to reduce the image size from 128 to 32
		3. Only 2 blocks were created and the final feature map was of 128 X 16 X 16

	2. Decoder : decoder which has 3 transpose convolution layers followed by a 1 X 1 which resulted in a 128X128 as output to expand the features to a mask
4. Loss function : MSE Loss
5. Total Parameters : 830,350

Further Work
The above arcihtecture was the initial model and we have to do a few changes based on the output
1. Create a background model that represents features of all the beaches which can be concatenated to the main model
2. Try out other loss functions
3. RGB Shift transformations to the images

Approach 2: Using a pretrained RESNET18 
1. Input to RESNET is 224 X 224 X 3 but we have an input of 224 X 224 X 6 
2. Applying a 1 X 1 convolution on the input to reduce the channels to 3
3. Exclude the last three layers from RESNET 18 fo the final feature from this model is 256 X 14 X 14
4. apply 4 transpose convolution layers followed by convolution layers to achieve an output of 224 X 224 X 1
5. Loss function : MSE Loss
6. Total Parameters : 402,996 (Trainable)


