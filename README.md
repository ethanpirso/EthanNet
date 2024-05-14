# EthanNet-30: A Deep Convolutional Neural Network for CIFAR-10

EthanNet-30 is a deep convolutional neural network designed for image classification tasks, specifically benchmarked on the CIFAR-10 dataset. The model integrates VGG-like blocks and ResNet-like bottleneck blocks to achieve effective feature extraction while maintaining a balance between depth and computational efficiency.

## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Custom Modules](#custom-modules)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Author](#author)

## Introduction

The aim of this project was to develop a better, smaller convolutional neural network (CNN) with good performance on the CIFAR-10 dataset. The model, EthanNet-30, combines the strengths of VGG and ResNet architectures to achieve high accuracy while being computationally efficient.

## Model Architecture

EthanNet-30 is structured as follows:
- **VGG Blocks**: Three VGG blocks with varying depths and increasing channels, each followed by optional dropout.
- **ResNet Block**: A DeepBottleneckResNet block that employs bottleneck modules with residual connections to enhance feature propagation without adding excessive parameters.
- **Pooling and Fully Connected Layers**: A final pooling layer to reduce spatial dimensions before classification, followed by two fully connected (FC) layers that condense the feature map into class predictions.
- **Regularization**: Batch normalization and dropout are employed post-pooling to stabilize and regularize the learning process.

## Custom Modules

### Bottleneck
A bottleneck layer that applies three convolutions with different kernel sizes and dilations, commonly used in deep residual networks. This module includes an optional residual connection that adapts to changes in input and output dimensions or strides.

### DeepBottleneckResNet
Constructs a sequence of bottleneck blocks, forming a deep bottleneck residual network. This module is suitable for constructing deep learning models that require efficient feature extraction with deep hierarchical structures.

### VGGBlock
A VGG-style block that sequentially applies multiple convolutions with ReLU activations, optionally followed by max pooling and dropout for regularization. This block is typically used for feature extraction in convolutional neural networks.

## Training Strategy

The training strategy involves:
- **Data Loading**: Using custom data loaders for the CIFAR-10 dataset with appropriate transformations.
- **Distributed Training**: Utilizing a custom data parallel strategy (`CustomDDPStrategy`) with the Gloo backend for GPU, as NCCL is not available on Windows machines. The training was conducted on NVIDIA RTX 3090 and RTX 4060 GPUs.
- **Early Stopping**: Implementing early stopping to prevent overfitting.
- **Precision**: Setting the default precision for matrix multiplication to 'medium' and using bf16-true precision during training.

## Results

*To be filled in later with the time it took to train, and the performance metrics (train/val/test accuracy, loss).*

## Author

Ethan Pirso
