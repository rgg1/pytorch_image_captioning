# CIFAR Image Classification with PyTorch

This is a machine learning project using PyTorch to construct a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset.

## Introduction

CIFAR-10 is a dataset of 50,000 32x32 color training images and 10,000 test images, categorized into 10 different classes. When a model is trained on this dataset, it can be used to predict the classes of new images.

## Installation

The following libraries are needed for this notebook:

* NumPy
* Matplotlib
* Torch
* Torchvision

If you're running this notebook on Google Colab, you should not have to manually install any libraries.

## Data Preparation

The CIFAR-10 data is loaded from the URL: https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz. It is then split into training and validation sets, with the training set further split into a smaller training set and a validation set.

Data loaders for training and validation are created, which allows iteration over the data in batches.

## Model Building

The model architecture is a Convolutional Neural Network (CNN). The layers of the CNN are as follows:

* Two Convolutional layers with a ReLU activation function followed by a MaxPool2d layer. This pattern repeats three times with an increasing number of output channels.
* A Flatten layer to transform the output into a single column tensor.
* Three linear (or Dense) layers with a ReLU activation function on the first two and no activation function on the last one.
The output from the model is a tensor of size 10 (the number of classes in the CIFAR-10 dataset).

## Model Training

The model is trained for 10 epochs using the Adam optimizer. Training and validation losses are calculated for each epoch, and the model parameters that result in the smallest validation loss are saved.

## Model Evaluation

The performance of the trained model is evaluated on the validation and test datasets, which were not seen during the training phase. This provides a measure of how well the model is likely to perform on unseen data. The validation and test accuracies were consistently over 75%.

## Prediction

Once the model is trained, it can be used to predict the class of new images. The 'predict_image' function can be used to predict the class of a single image.

## Conclusion

This notebook provides an implementation of a convolutional neural network using PyTorch for classifying images from the CIFAR-10 dataset. The model achieves a good accuracy on the validation and test sets. With further tuning of hyperparameters, data augmentation techniques, and potentially a more complex model, the accuracy could be improved.
