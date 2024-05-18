# EthanNet: A Deep Convolutional Neural Network for CIFAR-10

EthanNet is a deep convolutional neural network designed for image classification tasks, specifically benchmarked on the CIFAR-10 dataset. The model integrates VGG-like blocks and ResNet-like bottleneck blocks to achieve effective feature extraction while maintaining a balance between depth and computational efficiency.

## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Custom Modules](#custom-modules)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Author](#author)

## Introduction

The aim of this project was to develop a better, smaller convolutional neural network (CNN) with good performance on the CIFAR-10 dataset. The EthanNet models combine the strengths of VGG and ResNet architectures to achieve high accuracy while being computationally efficient. There are two versions of EthanNet: EthanNet-40 and EthanNet-39K.

## Model Architecture

### EthanNet-40
EthanNet-40 is structured as follows:
- **VGG Blocks**: Three VGG blocks with varying depths and increasing channels, each followed by optional dropout.
- **ResNet Block**: A DeepBottleneckResNet block that employs bottleneck modules with residual connections to enhance feature propagation without adding excessive parameters.
- **Pooling and Fully Connected Layers**: A final pooling layer to reduce spatial dimensions before classification, followed by two fully connected (FC) layers that condense the feature map into class predictions.
- **Regularization**: Batch normalization and dropout are employed post-pooling to stabilize and regularize the learning process.

### EthanNet-39K
EthanNet-39K is structured similarly to EthanNet-40 but features a Kolmogorov Arnold Network (KAN) layer instead of the final fully connected layers:
- **VGG Blocks**: Three VGG blocks with varying depths and increasing channels, each followed by optional dropout.
- **ResNet Block**: A DeepBottleneckResNet block that employs bottleneck modules with residual connections to enhance feature propagation without adding excessive parameters.
- **Pooling and KAN Layer**: A final pooling layer to reduce spatial dimensions before classification, followed by a KAN layer that maps the flattened ResNet block output to 10 classes.
- **Regularization**: Batch normalization and dropout are employed post-pooling to stabilize and regularize the learning process.

## Custom Modules

### Bottleneck
A bottleneck layer that applies three convolutions with different kernel sizes and dilations, commonly used in deep residual networks. This module includes an optional residual connection that adapts to changes in input and output dimensions or strides.

### DeepBottleneckResNet
Constructs a sequence of bottleneck blocks, forming a deep bottleneck residual network. This module is suitable for constructing deep learning models that require efficient feature extraction with deep hierarchical structures.

### VGGBlock
A VGG-style block that sequentially applies multiple convolutions with ReLU activations, optionally followed by max pooling and dropout for regularization. This block is typically used for feature extraction in convolutional neural networks.

### KANLayer
A Kolmogorov Arnold Network (KAN) layer that maps the flattened ResNet block output to class predictions. Unlike traditional fully connected layers that use linear transformations followed by activations at the nodes, the KAN layer employs learnable edges that are splines. This allows for more flexible and powerful function approximation, as the splines can adapt to the data in a non-linear fashion. In EthanNet-39K, the KAN layer provides an advanced alternative to fully connected layers, enhancing the model's ability to capture complex patterns in the data.

## Training Strategy

The training strategy involves:
- **Data Loading**: Using custom data loaders for the CIFAR-10 dataset with appropriate transformations.
- **Distributed Training**: Utilizing a custom data parallel strategy (`CustomDDPStrategy`) with the Gloo backend for GPU, as NCCL is not available on Windows machines. The training was conducted on NVIDIA RTX 3090 and RTX 4060 GPUs.
- **Early Stopping**: Implementing early stopping to prevent overfitting.
- **Precision**: Setting the default precision for matrix multiplication to 'medium' and using bf16-true precision during training.

## Results

EthanNet-40 training was halted after 10 epochs due to a lack of improvement in loss, triggering early stopping. This model achieved a disappointing accuracy of 29%. In contrast, EthanNet-39K, which shares a similar architecture, demonstrated significantly better performance. It trained for 40 epochs before early stopping was applied, achieving a test accuracy of 76%. While not the top-performing model, EthanNet-39K's efficiency is notable, thanks to its combination of VGG and ResNet architectures. The inclusion of the KAN layer, replacing the traditional fully connected layers, contributed to more efficient learning.

## Future Work

Future work should focus on evaluating the performance of models that exclusively use either VGG or ResNet architectures to determine their individual strengths and weaknesses. Additionally, the concept of Kolmogorov Arnold Network (KAN) layers shows promise and warrants further exploration. Investigating the feasibility of integrating KAN layers within convolutional layers could potentially lead to more powerful and flexible models. This approach could combine the benefits of convolutional operations with the advanced function approximation capabilities of KAN layers, potentially leading to significant improvements in model performance and efficiency.

## Authors

Ethan Pirso
