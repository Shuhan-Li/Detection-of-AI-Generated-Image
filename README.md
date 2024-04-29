# Detection-of-AI-Generated-Image

## Abstract
This project develops a CNN-based model to distinguish between AI-generated and real images. Utilizing the ResNet model for comparison and employing data augmentation and Bayesian hyperparameter optimization, this model achieves robust accuracy, aiding in the verification of image authenticity.

## Introduction
The advancement of AI has significantly impacted various domains, including digital content creation. However, this brings challenges, especially for content verification in the realm of art and photography. This project aims to address these challenges by developing a deep learning model to distinguish between AI-generated images and real photographs, supporting copyright protection and content verification applications.

## Background
CNN technology has evolved from its inception as LeNet to sophisticated architectures like AlexNet and R-CNN, leading today's AI capabilities in image recognition. This project utilizes these advanced CNN architectures to tackle the challenge of distinguishing AI-generated images in a reliable manner.

## Approach
CNN Architecture: Uses a sequential CNN architecture to identify hierarchical patterns in images, which includes multiple convolutional and resampling layers, and techniques to prevent overfitting like dropout.
ResNet Model: Employs the ResNet50 architecture, renowned for its deep network capabilities and residual learning framework, which helps in overcoming vanishing gradient issues during training.
Data Augmentation: Implements geometric transformations on images to increase data diversity and model robustness.
Bayesian Hyperparameter Tuning: Optimizes the CNN using Bayesian methods to efficiently search the hyperparameter space and improve model performance.
Early Stopping: Ensures the model does not overfit by stopping the training process once the performance on validation data ceases to improve.
Results
The model is trained on a dataset of 60,000 image pairs and tested on 10,000 pairs, achieving high accuracy in distinguishing between real and AI-generated images. The detailed results demonstrate the effectiveness of the CNN model, especially when enhanced by data augmentation and hyperparameter tuning.

## Dataset
The dataset consists of real images from the CIFAR-10 dataset and synthetic images generated using Stable Diffusion v1.4, providing a balanced set for training and validation. https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data

