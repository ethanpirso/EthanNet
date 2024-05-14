import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    """
    A bottleneck layer that applies three convolutions with different kernel sizes and dilations, 
    commonly used in deep residual networks. This module includes an optional residual connection that 
    adapts to changes in input and output dimensions or strides.

    Parameters:
        in_channels (int): Number of channels in the input tensor.
        mid_channels (int): Number of channels in the middle convolution layer.
        out_channels (int): Number of channels in the output tensor.
        dilation (int): Dilation rate for the middle convolution to increase the receptive field.
        stride (int): Stride used for the first convolution. Affects output size and optional residual connection.

    The first and third convolutions use a 1x1 kernel to adjust channel dimensions, while the second uses a 
    3x3 kernel. If the output dimensions or strides differ from the input, an additional convolution is applied 
    to the residual connection to match dimensions.
    """
    def __init__(self, in_channels, mid_channels, out_channels, dilation=1, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # If input and output dimensions are different, adjust residual path
        self.adjust_residual = (in_channels != out_channels) or (stride != 1)
        if self.adjust_residual:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.residual_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        if self.adjust_residual:
            residual = self.residual_bn(self.residual_conv(residual))

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += residual
        out = F.relu(out)
        return out
    
class DeepBottleneckResNet(nn.Module):
    """
    Constructs a sequence of bottleneck blocks, forming a deep bottleneck residual network. This module is 
    suitable for constructing deep learning models that require efficient feature extraction with deep 
    hierarchical structures.

    Parameters:
        in_channels (int): Number of channels in the input tensor to the first bottleneck block.
        block_sizes (list of int): List of output channel sizes for each block.
        repeat_n_times (list of int): List indicating how many times each block is repeated.
        dilation (int): Initial dilation rate, which can be increased across different blocks.

    Each block size in `block_sizes` corresponds to a bottleneck block with specified repeats, 
    potentially increasing the dilation rate for subsequent blocks to expand the receptive field progressively.
    """
    def __init__(self, in_channels, block_sizes, repeat_n_times, dilation=1):
        super(DeepBottleneckResNet, self).__init__()
        layers = []
        for idx, (out_channels, repeats) in enumerate(zip(block_sizes, repeat_n_times)):
            for _ in range(repeats):
                layers.append(Bottleneck(in_channels, out_channels // 4, out_channels, dilation=dilation))
                in_channels = out_channels  # Update in_channels for the next block
            if idx < len(block_sizes) - 1:  # Optionally add dilation increment here
                dilation *= 2
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class VGGBlock(nn.Module):
    """
    A VGG-style block that sequentially applies multiple convolutions with ReLU activations, optionally 
    followed by max pooling and dropout for regularization. This block is typically used for feature 
    extraction in convolutional neural networks.

    Parameters:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolutions.
        repeat (int): Number of times the convolution operations are repeated.
        pooling (bool): If True, applies max pooling after the convolutions.
        pool_size (int): Size of the window to take a max over for max pooling.
        dropout (float, optional): Dropout rate to apply after pooling (if pooling is True).

    The block enhances local feature extraction by applying several convolutions, each followed by 
    batch normalization and ReLU activation, to maintain the non-linearity in the model.
    """
    def __init__(self, in_channels, out_channels, repeat, pooling=True, pool_size=2, dropout=None):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(repeat):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=False))
            in_channels = out_channels  # Output becomes input for the next layer
        if pooling:
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
