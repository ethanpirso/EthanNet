import torch
import torch.nn as nn
from modules import VGGBlock
from kan import KANLayer

class EthanNet29K(nn.Module):
    def __init__(self):
        super(EthanNet29K, self).__init__()
        self.vgg_block1 = VGGBlock(3, 16, 2, dropout=0.2)
        self.vgg_block2 = VGGBlock(16, 32, 3, dropout=0.2)
        self.vgg_block3 = VGGBlock(32, 64, 3, dropout=0.2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.kan = KANLayer(64 * 2 * 2, 1, k=3)

    def forward(self, x):
        print("Forward pass started . . .")
        print(f"VGG Block1 input dtype: {x.dtype}")
        x = self.vgg_block1(x)
        print(f"VGG Block2 input dtype: {x.dtype}")
        x = self.vgg_block2(x)
        print(f"VGG Block3 input dtype: {x.dtype}")
        x = self.vgg_block3(x)
        print(f"Pool input dtype: {x.dtype}")
        x = self.pool(x)
        x = torch.flatten(x, 1)
        print(f"KAN input dtype: {x.dtype}")
        print(f"KAN input shape: {x.shape}")
        x, _, _, _ = self.kan(x)  # Unpacking all returned values
        print(f"Forward pass completed")
        return x

# Test Script
def test_model():
    print("-------------------------")
    print("Testing EthanNet29K model")
    print("-------------------------\n")
    model = EthanNet29K()
    print("--> Initializing model")
    print(f"--> Sending model to float32")
    model = model.to(torch.float32)
    print(f"--> Model dtype: {next(model.parameters()).dtype}\n")
    
    input_tensor = torch.randn(8, 3, 32, 32, dtype=torch.float32)  # Batch size 8, 3 channels, 32x32 image
    print(f"--> Created input tensor")
    print(f"--> Input dtype: {input_tensor.dtype}")
    print(f"--> Input shape: {input_tensor.shape}\n")
    
    print("--> Running forward pass")
    output = model(input_tensor)
    print(f"\n--> Output dtype: {output.dtype}")
    print(f"--> Output shape: {output.shape}\n")

if __name__ == "__main__":
    try:
        test_model()
        print("\n-------------------------")
        print("Tests passed successfully")
        print("-------------------------")
    except Exception as e:
        print("\n-------------------------")
        print("An error occurred during testing")
        print(f"Error: {str(e)}")
        print("-------------------------")
