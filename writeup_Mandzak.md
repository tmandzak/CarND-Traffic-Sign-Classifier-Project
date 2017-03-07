#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[random50signs.png]: ./examples/random50signs.png
[labelsFrequency.png]: ./examples/labelsFrequency.png
[minorityClassSign.png]: ./examples/minorityClassSign.png
[majorityClassSign.png]: ./examples/majorityClassSign.png
[grayscale.png]: ./examples/grayscale.png
[scaling.png]: ./examples/scaling.png
[perturbation.png]: ./examples/perturbation.png
[resampled.png]: ./examples/resampled.png
[0_Lenet_original.png]: ./examples/0_Lenet_original.png
[1_Lenet_Sigma_B128.png]: ./examples/1_Lenet_Sigma_B128.png
[2_Lenet_Sigma_Droput_0125.png]: ./examples/2_Lenet_Sigma_Droput_0125.png
[2_Lenet_Sigma_Droput_0125_0005_50.png]: ./examples/2_Lenet_Sigma_Droput_0125_0005_50.png
[2_Lenet_Sigma_Droput_0125_0005_50_under.png]: ./examples/2_Lenet_Sigma_Droput_0125_0005_50_under.png
[2_Lenet_Sigma_Droput_0125_0005_50_over.png]: ./examples/2_Lenet_Sigma_Droput_0125_0005_50_over.png
[2_Lenet_Sigma_Droput_0125_0005_50_over_perturb_0943.png]: ./examples/2_Lenet_Sigma_Droput_0125_0005_50_over_perturb_0943.png
[2_Lenet_Sigma_Droput_0125_0005_50_both_perturb_0919.png]: ./examples/2_Lenet_Sigma_Droput_0125_0005_50_both_perturb_0919.png
[3_Lenet_SD_G_perturb_stand_prelu_03_0973.png]: ./examples/3_Lenet_SD_G_perturb_stand_prelu_03_0973.png
[3_Lenet_SD_G_perturb_stand_prelu_03_max_0966.png]: ./examples/3_Lenet_SD_G_perturb_stand_prelu_03_max_0966.png
[3_Lenet_SD_G_perturb_stand_prelu_03_max_0972_0949_931_918_less_86_ms_FINAL.png]: ./examples/3_Lenet_SD_G_perturb_stand_prelu_03_max_0972_0949_931_918_less_86_ms_FINAL.png
[3_Lenet_SD_G_perturb_stand_prelu_03_max_0970_0951_937_934_less_86_ms_epoch100.png]: ./examples/3_Lenet_SD_G_perturb_stand_prelu_03_max_0970_0951_937_934_less_86_ms_epoch100.png
[3_Lenet_SD_G_perturb_stand_prelu_03_max_0976_0958_939_933_less_86_ms_epoch50_RGB.png]: ./examples/3_Lenet_SD_G_perturb_stand_prelu_03_max_0976_0958_939_933_less_86_ms_epoch50_RGB.png





[SermanetLeCun]: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/tmandzak/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

New images found on the web and rescaled to 32x32 sizes are placed in [signs](https://github.com/tmandzak/CarND-Traffic-Sign-Classifier-Project/tree/master/signs) folder. Other images used in this writeup can be found in [examples](https://github.com/tmandzak/CarND-Traffic-Sign-Classifier-Project/tree/master/examples) folder.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the code cell 3 of the IPython notebook.  

I used numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of training set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in code cells 5 - 10 of the IPython notebook.  
Method ```draw_signs``` (cell 5) is responsible for drawing a limited random subset of all signs or a limited sequential subset of a given class of signs. ```draw_signs``` was used to produce images below.

Following random set gives us general presentation of traffic sign images in various classes:

![alt text][random50signs.png]

Here is an exploratory visualization of the data set. It is a bar chart showing frequencies of sign classes in training, validations and test sets:

![alt text][labelsFrequency.png]

We can see here that classes are highly imbalanced so we need to try balancing them to reduce negative effects on the quality of classification. Cell 8 gives us mean frequency of 809 and we'll use it as a target for random under- and oversampling (augmentation). We can also see that classes in test and validation sets are distributed the same way as in the training set. 

Following two figures present sequential (as met in the dataset) subsets of signs in minority and majority classes correspondingly:

Minority class
![alt text][minorityClassSign.png]

Majority class
![alt text][majorityClassSign.png]

For now I'll just notice that both classes contain groups of relatevely similiar images that are probably obtained from video frames. 

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Conversion to grayscale is a widely used technique in Computer Vision that lets reduce model complexity so I've decided to apply it in this project as well. Another technique I am going to try out is [Histogram equalization](https://en.wikipedia.org/wiki/Histogram_equalization). This allows for areas of lower local contrast to gain a higher contrast. Demonstration of these methods is performed in cells **12-16** and includes following illustration: 

![alt text][grayscale.png]

Variety in the range of values of raw data may cause objective functions to work inproperly without normalization. In this project I'm going to give a try to these three ways of normalization:

* scale the range in [−1, 1] - (squeeze)
* standardize 
* scale the range in [0, 1] - (normalize)

All three are implemented and demonstrated in cells **20-21**. Here is an example of a traffic sign image before and after normalization:

![alt text][scaling.png]

Full preprocessing is implemented in the cell **22** (method ```preprocess```) and is applied to the dataset in the cell **37**.
Note that preprocessing also includes augmentation of the training data mentioned later.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

In this project I just used the validation set provided in valid.p file.

Code cells 17-19 of the IPython notebook contain the code for balancing training data classes through random under- and oversampling with options to just copy existing images or apply additional random perturbation. I decided to try out undersampling of majority classes in the training dataset since there can be lots of similarities in majority classes as demonstrated on exploratory visualisation figure. Images in minority classes can be randomly perturbed in position ([-2,2] pixels), in scale ([.9,1.1] ratio) and rotation ([-15,+15] degrees) during augmentation as suggested in Sermanet & LeCun [article][SermanetLeCun].

Here is an example of aforementioned perturbations:

![alt text][perturbation.png]

I am going to try out following cases of balancing:

* undersampling, when majority classes are decreased to 809 members by random drop out
* oversampling, when minority classes are increased to 809 members by random duplicationg
* oversampling with perturbation (augmentation)
* undersampling + oversampling simulaneously, when all classes have equal amounts of 809 members

Balancing is implemented in the ```resample``` method that is applied to training dataset in the cell **37** inside of the ```preprocess``` method as noted before.

After augmentation of training data freaquencies of classes look like this:

![alt text][resampled.png]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the cell **31** of the ipython notebook in the ```LeNet_multiscale``` method. 
The model represents Convolutional Neural Network that implements the idea of Multi-Scale Features proposed by [Sermanet & LeCun][SermanetLeCun] and is built iteratively starting from original LeNet-5 architecture.
All batch and channel strides are 1, all paddings are 'VALID'.
My final model consists of the following layers:

| Layer         		|     Description	        				                    	| 
|:---------------:|:------------------------------------------------:| 
| Input         		| 32x32xL, L=3 for RGB and L=1 for Grayscale image | 
| Convolutional 1 | 5x5xLx6, 1x1 stride, outputs 28x28x6  	          |
| Max pooling	1  	| 2x2 filter, 2x2 stride, outputs 14x14x6          |
| Activation 1    |	PReLU                                            |
| Convolutional 2 | 5x5x6x6, 1x1 stride, outputs 10x10x16 	          |
| Max pooling	2  	| 2x2 filter, 2x2 stride, outputs 5x5x16           |
| Activation 2    |	PReLU                                            |
| Flatten 1       | applied to Activation 2 5x5x16, ouputs 400       |
| Avg pooling     | applied to Activation 1 14x14x6, ouputs 7x7x6    |
| Flatten 2       | applied to Avg pooling 7x7x6, outputs 294        |
| Concatenation   | Flatten 1 + Flatten 2, ouputs 694                |
| Fully connected	| input 694, outputs 86						                      |
| Activation 3    |	PReLU                                            |
| Dropout         |                                                  |
| Fully connected	| input 86, outputs 43 						                      |


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in cells **33-40** of the ipython notebook. 
To train the model, I used the code from [LeNet Lab Solution](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb) that was slightly modified and reorganized to a few mothods:

* ```evaluate``` - responsible for evaluation as in the original code
* ```train_and_validate``` - responsible for running training and validation
* ```plot_eval_dict``` - responsible for plotting Loss vs. Epoch and Training/Validation Accuracy vs. Epoch from the output of  ```train_and_validate```
* hyperparameters initialization in the cell **36**
* preprocessing including augmentation in the cell **37**
* plotting frequencies of classes in training data set before and after augmentation in the cell **38**
* training data shuffling in the cell **39**
* cells **25 - 31** contain a few models developed on the way from original LeNet-5 to final Lenet_multiscale that is executed in the cell **40**. These models are placed in brackets when mentioned below (```LeNet_original``` ```LeNet_sigma_dropout``` ```LeNet_max``` ```LeNet_short``` ```LeNet_multiscale```). 

Hyperparameters for the final model were set as follows (cell **36**):
```
EPOCHS = 50
BATCH_SIZE = 128
rate = 0.0005
mu = 0
sigma = 0.1  # used only for original LeNet
keep_p = 0.125  # dropout parameter
prelu_alpha = 0.3 # PreLu parameter
```

Parameter ```sigma``` will be calculated inside of ```LeCun_multiscale``` method separately for each layer depending on the number of inputs as suggested by [Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](https://arxiv.org/abs/1502.01852).
```sigma = np.sqrt(2/n)``` for layers with PreLu activation and as ```sigma = np.sqrt(1/n)``` for layers without (last one), where ```n``` is a number of inputs. This aproach helps to improve initialization as will be shown later.

Parameters of the preprocessing can be found in the cell **37**:

```
grayscale=True    # use Grayscale conversion
eqHist=False      # don't use Histogram Equalization
undersample=False # don't use undersampling
oversample=True   # use oversampling...
perturb=True      # ...with perturbations (translate, scale, rotate)
rescaling='standardize' # use Standardization for normalization
```

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating training and validation accuracy of the model is located inside of ```train_and_validate``` method (cell **34**) that reuses ```evaluation``` method defined in previous cell **33**. 
Test accuracy is calculated in the cell **41** calling the same ```evaluation``` method.
In the next cell average Precision and Recall are obtained.

My final model results were:
* training set accuracy of **0.995**
* validation set accuracy of **0.972**
* test set accuracy of **0.949**
* average Precision of **0.931**
* average Recall of **0.918**

I've chosen an iterative approach starting from original LeNet-5 model  as a familiar and working example.
First I run it (```LeNet_original``` model) on RGB images without any preprocessing and augmentation with this set of parameters:

```
EPOCHS = 20
BATCH_SIZE = 128
rate = 0.001
mu = 0
sigma = 0.1
```
and got following results:

![alt text][0_Lenet_original.png]

Though the learning rate looks to be good as it can be seen from Loss plot, the validation accuracy on starting epochs is below 0.5 and there is quite big gap between training and validation accuracy telling us about overfitting.
On applying ```sigma``` formulas mentioned above we got an emmidiate improvement of initialization (```LeNet_sigma_dropout``` model):

![alt text][1_Lenet_Sigma_B128.png]

Trying ```BATCH_SIZE = 64``` and ```BATCH_SIZE = 256``` wasn't successfull, so I left it to be ```128```.
To reduce overfitting I applied dropout with ```keep_p = 0.125``` (```LeNet_sigma_dropout``` model), but it was not enough since Loss plot indicated bad learning rate and plots of accuracies tell us to increase number of ```EPOCHS```:

![alt text][2_Lenet_Sigma_Droput_0125.png]

Using this set of parameters worked well from now on:

```
EPOCHS = 50
BATCH_SIZE = 128
rate = 0.0005
mu = 0
keep_p = 0.125
```

![alt text][2_Lenet_Sigma_Droput_0125_0005_50.png]

Next step was to try random undersampling, oversampling and oversampling with perturbations:

Undersampling

![alt text][2_Lenet_Sigma_Droput_0125_0005_50_under.png]

Oversampling

![alt text][2_Lenet_Sigma_Droput_0125_0005_50_over.png]

Oversampling with perturbations

![alt text][2_Lenet_Sigma_Droput_0125_0005_50_over_perturb_0943.png]

Both Undersampling and Oversampling with perturbations:

![alt text][2_Lenet_Sigma_Droput_0125_0005_50_both_perturb_0919.png]

And as it can be easely seen, oversampling with perturbations gave us the best result with 0.943 Validation Accuracy.

Next using Grayscale conversion with Standardization normalization gave us slightly better Validation Accuracy of 0.966 comparing to 0.959 for [0, 1] scaling and 0.961 for [-1, 1] scaling. Since Histogram Equalization didn't improve this accuracy I didn't use it for later experiments.

Original LeNet model already uses RELU activation, but I was curious what if I substitute it with a more general PreLU? 
Tweaking the only PreLu parameter ```prelu_alpha``` let us achieve even 0.973 Validation Accurracy for ```prelu_alpha = 0.3``` (```LeNet_prelu``` model):

![alt text][3_Lenet_SD_G_perturb_stand_prelu_03_0973.png]

Another thing to try was placing pooling before PreLU so that PreLU deals with smaller input as recommended by [Nikolas Markou](http://nmarkou.blogspot.ca/2017/02/the-black-magic-of-deep-learning-tips.html?utm_content=bufferab398&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer&m=1). Though numbers inputs for PreLUs decreased 4 times it had no visible impact on accuracy (```LeNet_max``` model):

![alt text][3_Lenet_SD_G_perturb_stand_prelu_03_max_0966.png]

Since I suspected the model to be still overfitted I decided to experiment simplifing the model. The most successfull try was to remove middle fully connected layer ```fc2``` and set the number of outputs for  ```fc1``` to be twice as number of classes or 86. Such a reduction still didn't influence the Valdation Accuracy (```LeNet_short``` model).

The last modification of the model was implementation of an idea of Multi-Scale feature mentioned by [Sermanet & LeCun][SermanetLeCun]. Here I've just concatenated flattened outputs of the second convolution layer and the first one previously subsampled by average pooling. This update let me achive the result of **0.972 Validation Accuracy** and **0.949 Test Accuracy**:

![alt text][3_Lenet_SD_G_perturb_stand_prelu_03_max_0972_0949_931_918_less_86_ms_FINAL.png]

Last two experiments were to increase number of epochs to 100:

![alt text][3_Lenet_SD_G_perturb_stand_prelu_03_max_0970_0951_937_934_less_86_ms_epoch100.png]

and to use color images as an input:

![alt text][3_Lenet_SD_G_perturb_stand_prelu_03_max_0976_0958_939_933_less_86_ms_epoch50_RGB.png]

As a result slightly better **Test Accuracies** were achieved: **0.951** and **0.958** respectively.
If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


