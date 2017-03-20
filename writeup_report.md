#**Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4


###Model Architecture and Training Strategy
####2. Data that was used to train the model
at first I used the data that was offered by Udaciy.
However the recorded images also pictured the sky, trees and the hood of the car. Therefore the first step was to crop 70px from the top and 25px from the bottom.
My second step was to normalize the images.
Here is an original image followed by and cropped and normalized image:

![original](https://github.com/christianreiser/P3-Behavioral-Cloning/blob/master/Images/c.jpg)

![](https://github.com/christianreiser/P3-Behavioral-Cloning/blob/master/Images/normalized.png)

 However my model always got stuck on the same curves. Consequently I drove in training mode and recorded more data. Especially on the difficult curves I turned recording on, drove the curve, turned recording off, drove back and started with the same curve again. This helped me alot.
Another problem was when the car got too far to the side, it could not hadle it anymore. Thats why I also used the camera on the left and on the right. I implemented the camera on the left&right and adjusted the steering angle by +0.25 & -0.25. 
Here are images from the left, center and right:

![left](https://github.com/christianreiser/P3-Behavioral-Cloning/blob/master/Images/l.jpg)
![center](https://github.com/christianreiser/P3-Behavioral-Cloning/blob/master/Images/c.jpg)
![right](https://github.com/christianreiser/P3-Behavioral-Cloning/blob/master/Images/r.jpg)

Later on I recognized that the car was pulling to the left, and I read this is due to driving in a circle counterclockwise. So I argumented my dataset by flippung the images with a probability of 0.5 on the vertical axis and multiplied the steering angles by -1.
Here are two images, where one is fliped.

![](https://github.com/christianreiser/P3-Behavioral-Cloning/blob/master/Images/normalized.png)

![flipped](https://github.com/christianreiser/P3-Behavioral-Cloning/blob/master/Images/fliped%20the%20normalized.png)

Afer training with the argumented dataset the car pulled too far to the right and I'm not sure why. The only way I was able to fix this issue was by manipulating the steering angle by -0.31.


####3. An appropriate model architecture has been employed

My first step was to use a convolution neural network model similar to  [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py)'s
I thought this model might be appropriate because it used Convolutions as the first layers, which is very usefuly to reduce the numer of parameters.
It also contained dropout wich was very helpful in order to reduce overfitting (more details below).
The last Layers are fully connected ones so my model is able to recognize details.

The architecture is:
0. randomly shuffle the data set 
1. Cropping
2. normalization
3. Convolution (16x8x8; subsample:4x4; Padding:same
4. Relu activation
5. Convolution (32x5x5; subsample:2x2; Padding:same
6. Relu activation
7. Convolution (64x5x5; subsample:2x2; Padding:same
8. Flatten
9. Dropout:0.2
10. Relu activation
11. Fully connected layer:512
12. Dropout:0.5
13. Relu activation
14. Fully connected layer:1

I used the Adam optimizer with a learningrate of 0.0001 and mse as the loss function

### 4. Model Training Strategy


To see if my model is working at all, I started with only a few images and one epoche. 
Later I used more data and increased the number of epoches. Also I split my image and steering angle data into a training and validation set. At earlier with a ratio of 0.3. Later on to provide more data to train I changed it to a ratio of 0.1.
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set right after the first epoche. This implied that the model was overfitting.

### Attempts to reduce overfitting in the model
Overfitting was a big problem. My attempts to reduce overfitting was adding dropout, adding moredata, reducing the rearningrate and setting the number of epoches to just one.
The model contains dropout layers in order to reduce overfitting. 
After the collection process, I had 24108 data points. 



### 5. Conclusion
This project was difficult at first because the car was not driving well at all. Later on it was fun and I learned a lot:

#### Things I learned:
1. Keras is a really fast war to try different architectures quickly.
2. Training with only one epoche can be enough
3. GPU enabled training is much faster
4. Argumenting the data at critical locations helpes very much.
5. Croping images so unuseful parts are not shown gives better results and leads to a faster trainingprocess.
6. When stuck look at open sourced pipelines like comma.ai's ;)


