## Traffic Sign Recognition

The goals of this project were the following:

* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./examples/visualize.png 
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/ts10.jpeg "General caution"
[image4]: ./examples/ts2.jpeg "Yield"
[image5]: ./examples/ts3.jpeg "Right-of-way at the next intersection"
[image6]: ./examples/ts6.jpeg "Vehicles over 3.5 metric tons prohibited"
[image7]: ./examples/ts7.jpeg "Turn right ahead"
[image8]: ./examples/hist_train.png
[image9]: ./examples/hist_valid.png
[image10]: ./examples/hist_test.png
[image11]: ./examples/before_process.png "Original traffic sign images"
[image12]: ./examples/after_process.png "Grayscaled images"


### Overview

In this project I used deep neural networks and convolutional neural networks to build a model that was then used to classify traffic signs. The model trained and validated traffic sign images from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model was trained it was then used to classify traffic sign images never before seen by the model.


### Results

The image classifiction model was built using Python, Here is a link to my [project code](Traffic_Sign_Classifier.py). I used the NumPy library to calculate summary statistics of the traffic signs data set:

* The size of training set was = 34799
* The size of the validation set was = 4410
* The size of test set was = 12630
* The shape of a traffic sign image was = (32, 32, 3)
* The number of unique classes/labels in the data set was = 43

Here is an exploratory visualization of the data set. The image shown below demonstrates its size, shape and classification which is captured for all images in the three data sets. There are three(3) bar charts showing how the image data is distributed over the 43 traffic sign classes being explored in the Train, Validation and Test sets.

![alt_text][image1] 

![alt_text][image8]
![alt_text][image9]
![alt_text][image10]

As a first step, I decided to convert the images to grayscale because the color images were of poor quality and grayscale  accentuated the identifiable features of the traffic signs making them more discernable. This allowed for better classification of the traffic signs. The improved contrast between the traffic signs and their backgrounds is most obvious for images 2,4,6,7 and 8 in the before and after example images shown below. Initially I started training the model with the color images but the validation accuracies were lower than expected.

Here is an example of traffic sign images before and after grayscaling.

![alt text][image11]
![alt_text][image12]

As a last step, I normalized the image data because it created well conditioned input values with zero mean and equal variance which helps to ensure that the value obtained from the loss function is neither too big nor too small. In general, conditioned input values reduce the searching required of an optimizer, when solving problems, thus making it easier and quicker in this case to classify the traffic signs. 


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image 						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout               |                                               |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Dropout				|												| 
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Fully connected		| input 400,   output 120						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 120,   output 84						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 84,   output 43  						|
| Softmax				| input 43,   output 43							|
|						|												|
|						|												|
 

To train the model, I used the AdamOptimizer with a batch size of 128 images at a time from the training set of 34,799 images in total, then I ran 20 forward and backward propagations on each batch using 20 EPOCHS. The hyperparameters chosen for the normal distribution to generate the initial weights of each convolutional layer was (mean = 0 ; standard deviation = 0.1). The learning rate for the optimizer, which is the rate at which the weights were adjusted so as to minimize the error/ loss function was set to (rate = 0.001). The error is captured in the cross_entropy variable which is a measure of how accurately the output of the model corresponds with the ground truth labels initially given to the model.


The LeNet architecture was used to create this model, it has been used for character recognition in the past not only in academia but also adopted in industries eg.the USPS for reading handwritten address/zipcodes and Banks to read the MICR encoded on checks. The architecture lends itself well to the problem of traffic sign classification, because of the use of convolution it is possible to detect relationships between adjacent pixels in an image which improves the ability to determine the subject/content of that image. It is also possible to add layers to the model which increases its depth allowing it to classify images based on many more features.

For this model the subsampling layers of the original LeNet architecture were removed and Pooling layers added. Additionally, Dropout layers were added to reduce the observed overfitting at the training stage, a keep_prob = 0.75 was used in the model for this stage.

The relatively small variation in accuracies between the three stages indicated that the model was working as desired.

My final model results were:
* training set accuracy of: 0.996 
* validation set accuracy of : 0.955 
* test set accuracy of: 0.932


In order to test the accuracy of the model on new images I loaded five German traffic signs that were not seen before.
The five images are shown below.

![alt text][image7] ![alt text][image5] ![alt text][image4] 
![alt text][image3] ![alt text][image6]

The denoted traffic sign in each image was clear of any optical defects and for most images the traffic sign took up a relatively large portion of the image and/or was the most significant feature of the image except for the fourth image where the sign was comparably smaller and had other contending features in the image. As a result this traffic sign was misclassified.

Here are the results of the prediction:

| Image												| Prediction									| 
|:-------------------------------------------------:|:---------------------------------------------:| 
| Turn right ahead									| Turn right ahead								| 
| Right-of-way at the next intersection 			| Right-of-way at the next intersection			|
| Yield												| Yield											|
| General caution									| General caution				 				|
| Vehicles over 3.5 metric tons prohibited			| Roundabout mandatory 							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on a much larger test set of 93.8%

For the first image, the model is very confident that this was a Turn right sign (probability of 0.99), and the image does contain a Turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Turn right ahead								| 
| .000048  				| Ahead only									|
| .000018				| Go straight or left							|
| .0000037    			| Right-of-way at the next intersection			|
| .0000033			    | Speed limit (30km/h) 							|


For the second image the model was very confident that this was a Right-of-way at the next intersection sign (probability of 0.99), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were 



| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Right-of-way at the next intersection			| 
| .0000023 				| Beware of ice/snow							|
| .000000005			| Pedestrians   								|
| .0000000039  			| General caution				 				|
| .0000000031		    | Roundabout mandatory 							|

For the third mage the model was very confident that this was a Yield sign (probability of 0.99), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Yield     									| 
| .000071  				| Ahead only									|
| .000061				| Priority road									|
| .000015      			| Slippery road					 				|
| .00000063			    | No vehicles       							|


For the fourth image the model was very confident that this was a General caution sign (probability of 0.989), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .989         			| General caution   							| 
| .0080    				| Right-of-way at the next intersection    		|
| .00104				| Speed limit (30km/h)							|
| .00109      			| Turn right ahead				 				|
| .000018			    | Beware of ice/snow   							|

For the fifth image the model was very unsure that this was a Turn right sign (probability of 0.257), and the image does NOT contain a Roundabout mandatory sign. The top five soft max probabilities were

| Probability         	|     Prediction	            					| 
|:---------------------:|:-------------------------------------------------:| 
| .257         			| Roundabout mandatory      						| 
| .169     				| Keep right 										|
| .122					| No passing for vehicles over 3.5 metric tons		|
| .090	      			| End of no passing by vehicles over 3.5 metric tons|
| .076				    | No passing            							|








## Udacity's original README
Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

