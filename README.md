Overview
This repository contains two projects focused on image classification using neural networks, implemented with the Keras library. These projects demonstrate the use of fully connected and convolutional neural networks (CNNs) to classify datasets, preprocess data, and evaluate model performance.

The projects cover the following datasets:

MNIST Handwritten Digits Dataset
CIFAR-10 Image Dataset
Project 1: MNIST Handwritten Digit Classification
Description
This project trains a fully connected neural network (multilayer perceptron) to classify grayscale images of handwritten digits from the MNIST dataset. The dataset contains 60,000 training images and 10,000 test images, each of size 28x28 pixels.

Key Features
Dataset: MNIST (10 classes: digits 0–9)
Model Architecture:
Two hidden layers with 128 neurons each
ReLU activation for hidden layers
Softmax activation for the output layer
Optimizer: Stochastic Gradient Descent (SGD)
Performance Metric: Accuracy
File: mnist_classification.py
Workflow
Data Loading and Preprocessing:
Load and split MNIST dataset into training and testing sets.
Normalize input data to the range [0, 1].
One-hot encode class labels.
Model Architecture:
Fully connected feed-forward neural network with two hidden layers and an output layer.
Training:
Train the model with a validation split for 20 epochs.
Evaluation:
Test the trained model on unseen test data and report accuracy and loss.
Project 2: CIFAR-10 Image Classification Using CNNs
Description
This project trains a convolutional neural network (CNN) to classify color images from the CIFAR-10 dataset. The dataset consists of 50,000 training images and 10,000 test images, each of size 32x32 pixels with 3 color channels.

Key Features
Dataset: CIFAR-10 (10 classes: airplane, car, bird, cat, etc.)
Model Architecture:
Two convolutional layers with max pooling and ReLU activation
Dropout layers for regularization
Fully connected layers with 512 neurons and softmax activation
Data Augmentation:
Rotation, shifting, zooming, and horizontal flipping
Optimizer: RMSprop
Performance Metric: Accuracy
File: cifar10_classification.py
Workflow
Data Loading and Preprocessing:
Load and split CIFAR-10 dataset into training and testing sets.
Normalize input data to the range [0, 1].
One-hot encode class labels.
Model Architecture:
CNN with two convolutional layers, max pooling, dropout, and fully connected layers.
Data Augmentation:
Use ImageDataGenerator for real-time augmentation during training.

Training:
Train the model using augmented data for 20 epochs.

Evaluation:
Test the trained model on unseen test data and report accuracy and loss.
Requirements
Python 3.7+
Libraries:
keras
numpy
matplotlib
ssl (for handling secure dataset downloads)
Ensure GPU support for faster training if available.

Installation
Clone the repository:
bash
Copy code
git clone https://github.com/your-repo-url.git
cd your-repo-url

Install dependencies:
bash
Copy code
pip install -r requirements.txt
Usage
Run MNIST Classification
bash
Copy code
python mnist_classification.py
Run CIFAR-10 Classification
bash
Copy code
python cifar10_classification.py
Results
MNIST Project
Test Accuracy: ~94.6%
Fully connected neural network effectively learns from grayscale digit images.
CIFAR-10 Project
Test Accuracy: ~92.5%
CNN with data augmentation achieves robust performance on small color images.
Folder Structure
graphql
Copy code
.
├── mnist_classification.py        # MNIST digit classification script
├── cifar10_classification.py      # CIFAR-10 image classification script
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
Notes
Experiment with hyperparameters like batch size, number of layers, or dropout rate to further optimize model performance.
GPU acceleration can significantly speed up training, especially for the CIFAR-10 project.
License
This project is licensed under the MIT License. See LICENSE for details.

Author
Conor Steward
1conorsteward@gmail.com






