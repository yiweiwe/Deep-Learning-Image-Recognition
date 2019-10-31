# Deep-Learning-Image-Recognition

Can a machine identify a bee as a honey bee or a bumble bee? These bees have different behaviors and appearances, but given the variety of backgrounds, positions, and image resolutions, it can be a challenge for machines to tell them apart. Being able to identify bee species from images is a task that ultimately would allow researchers to more quickly and effectively collect field data. Pollinating bees have critical roles in both ecology and agriculture, and diseases like colony collapse disorder threaten these species. Identifying different species of bees in the wild means that we can better understand the prevalence and growth of these important insects.

![image](https://user-images.githubusercontent.com/46982385/67970509-7c1d0200-fbe1-11e9-95d0-ece135f75e4a.png)

This notebook walks through building a simple deep learning model that can automatically detect honey bees and bumble bees and then loads a pre-trained model for evaluation.

### Load image labels
Now that we have all of our imports ready, it is time to look at the labels for our data. We will load our labels.csv file into a DataFrame called labels, where the index is the image name (e.g. an index of 1036 refers to an image named 1036.jpg) and the genus column tells us the bee type. genus takes the value of either 0.0 (Apis or honey bee) or 1.0 (Bombus or bumble bee).

### Examine RGB values in an image matrix
Image data can be represented as a matrix. The width of the matrix is the width of the image, the height of the matrix is the height of the image, and the depth of the matrix is the number of channels. Most image formats have three color channels: red, green, and blue.
For each pixel in an image, there is a value for every channel. The combination of the three values corresponds to the color, as per the RGB color model. Values for each color can range from 0 to 255, so a purely blue pixel would show up as (0, 0, 255).

![image](https://user-images.githubusercontent.com/46982385/67970806-fea5c180-fbe1-11e9-9d06-4b083a38e3cd.png)

### Normalize image data
Now we need to normalize our image data. Normalization is a general term that means changing the scale of our data so it is consistent.
In this case, we want each feature to have a similar range so our neural network can learn effectively across all the features. As explained in the sklearn docs, "If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected."
We will scale our data so that it has a mean of 0 and standard deviation of 1. We'll use sklearn's StandardScaler to do the math for us, which entails taking each value, subtracting the mean, and then dividing by the standard deviation. We need to do this for each color channel (i.e. each feature) individually.
### Split into train, test, and evaluation sets
Now that we have our big image data matrix, X, as well as our labels, y, we can split our data into train, test, and evaluation sets. To do this, we'll first allocate 20% of the data into our evaluation, or holdout, set. This is data that the model never sees during training and will be used to score our trained model.
We will then split the remaining data, 60/40, into train and test sets just like in supervised machine learning models. We will pass both the train and test sets into the neural network.
### Model building (part i)
It's time to start building our deep learning model, a convolutional neural network (CNN). CNNs are a specific kind of artificial neural network that is very effective for image classification because they are able to take into account the spatial coherence of the image, i.e., that pixels close to each other are often related.
Building a CNN begins with specifying the model type. In our case, we'll use a Sequential model, which is a linear stack of layers. We'll then add two convolutional layers. To understand convolutional layers, imagine a flashlight being shown over the top left corner of the image and slowly sliding across all the areas of the image, moving across the image in the same way your eyes move across words on a page. Convolutional layers pass a kernel (a sliding window) over the image and perform element-wise matrix multiplication between the kernel values and the pixel values in the image.
### Model building (part ii)
Let's continue building our model. So far our model has two convolutional layers. However, those are not the only layers that we need to perform our task. A complete neural network architecture will have a number of other layers that are designed to play a specific role in the overall functioning of the network. Much deep learning research is about how to structure these layers into coherent systems.
We'll add the following layers:
MaxPooling. This passes a (2, 2) moving window over the image and downscales the image by outputting the maximum value within the window.
Conv2D. This adds a third convolutional layer since deeper models, i.e. models with more convolutional layers, are better able to learn features from images.
Dropout. This prevents the model from overfitting, i.e. perfectly remembering each image, by randomly setting 25% of the input units to 0 at each update during training.
Flatten. As its name suggests, this flattens the output from the convolutional part of the CNN into a one-dimensional feature vector which can be passed into the following fully connected layers.
Dense. Fully connected layer where every input is connected to every output (see image below).
Dropout. Another dropout layer to safeguard against overfitting, this time with a rate of 50%.
Dense. Final layer which calculates the probability the image is either a bumble bee or honey bee.
To take a look at how it all stacks up, we'll print the model summary. Notice that our model has a whopping 3,669,249 paramaters. These are the different weights that the model learns through training and what are used to generate predictions on a new image.

![image](https://user-images.githubusercontent.com/46982385/67970958-488ea780-fbe2-11e9-8d12-1d79892c1aa0.png)

### Compile and train model
Now that we've specified the model architecture, we will compile the model for training. For this we need to specify the loss function (what we're trying to minimize), the optimizer (how we want to go about minimizing the loss), and the metric (how we'll judge the performance of the model).
Then, we'll call .fit to begin the trainig the process.
"Neural networks are trained iteratively using optimization techniques like gradient descent. After each cycle of training, an error metric is calculated based on the difference between prediction and target...Each neuron’s coefficients (weights) are then adjusted relative to how much they contributed to the total error. This process is repeated iteratively." ML Cheatsheet
Since training is computationally intensive, we'll do a 'mock' training to get the feel for it, using just the first 10 images in the train and test sets and training for just 5 epochs. Epochs refer to the number of iterations over the data. Typically, neural networks will train for hundreds if not thousands of epochs.
Take a look at the printout for each epoch and note the loss on the train set (loss), the accuracy on the train set (acc), and loss on the test set (val_loss) and the accuracy on the test set (val_acc). We'll explore this more in a later step.
### Load pre-trained model and score
Now we'll load a pre-trained model that has the architecture we specified above and was trained for 200 epochs on the full train and test sets we created above.
Let's use the evaluate method to see how well the model did at classifying bumble bees and honey bees for the test and validation sets. Recall that accuracy is the number of correct predictions divided by the total number of predictions. Given that our classes are balanced, a model that predicts 1.0 for every image would get an accuracy around 0.5.
Note: it may take a few seconds to load the model. Recall that our model has over 3 million parameters (weights), which are what's being loaded.
### Visualize model training history
In addition to scoring the final iteration of the pre-trained model as we just did, we can also see the evolution of scores throughout training thanks to the History object. We'll use the pickle library to load the model history and then plot it.
Notice how the accuracy improves over time, eventually leveling off. Correspondingly, the loss decreases over time. Plots like these can help diagnose overfitting. If we had seen an upward curve in the validation loss as times goes on (a U shape in the plot), we'd suspect that the model was starting to memorize the test set and would not generalize well to new data.

![image](https://user-images.githubusercontent.com/46982385/67971059-77a51900-fbe2-11e9-80a8-08e179f52817.png)

### Generate predictions
Previously, we calculated an overall score for our pre-trained model on the validation set. To end this notebook, let's access probabilities and class predictions for individual images using the .predict and .predict_classes methods.
We now have a deep learning model that can be used to identify honey bees and bumble bees in images! The next step is to explore transfer learning, which harnesses the prediction power of models that have been trained on far more images than the mere 1600 in our dataset.

![image](https://user-images.githubusercontent.com/46982385/67971100-88ee2580-fbe2-11e9-931d-f12ac99d2234.png)
