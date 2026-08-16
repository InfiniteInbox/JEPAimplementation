import torch.nn as nn
import torch
# Using a Conv NN due to low data application and task of next frame prediction instead of using input masking
# We still use Exponantial Moving Averages, and Student Teacher frameworks for training to stay faithful to the V-JEPA paper
# As a result we create an embedding for the entire image, instead of each portion (kernel)
class encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, 64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1)
        self.norm3 = nn.GroupNorm(8, 128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=2, padding=1)
        self.norm4 = nn.GroupNorm(8, 256)
        #256 x 4 x 4
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(4096, 128)
        #128 Dim Embedding 
    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.activation(self.norm3(self.conv3(x)))
        x = self.activation(self.norm4(self.conv4(x)))
        x = self.flatten(x)
        x = self.lin1(x)
        #No Loss as the GELU prevents Isotropic Gaussian (Important for LeWorldModel Arch)
        return x

'''
a = encoder()
print(a.forward(torch.zeros(2, 4, 64, 64)).shape)
'''
class predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(128, 256)
        self.activate = nn.GELU()
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 128)
    def forward(self, x):
        x = self.activate(self.layer1(x))
        x = self.activate(self.layer2(x))
        x = self.layer3(x)
        return x
'''
a = predictor()
print(a.forward(torch.zeros(2, 128)).shape)
'''