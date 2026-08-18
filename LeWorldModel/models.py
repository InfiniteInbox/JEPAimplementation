import torch.nn as nn
import torch

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
        #No Loss as the GELU prevents Isotropic Gaussian (Important for LeWorldModel Arch), my foresight is almost too insane ;)
        return x
# Beefed up encoder as we are dealing with action states
class predictor(nn.Module):
    def __init__(self, latent=128, n_actions=25, n_dim=48, hidden=1024):
          super().__init__()
          self.action_embedding = nn.Embedding(n_actions, n_dim)
          self.linear1 = nn.Linear(latent + n_dim, hidden)
          self.loss = nn.GELU() # GELU in the predictor is fine
          self.linear2 = nn.Linear(hidden, hidden//2)
          self.linear3 = nn.Linear(hidden//2, latent)
    def forward(self, z, a):
        e = self.action_embedding(a)
        x = torch.cat([z,e], dim=1)
        x = self.loss(self.linear1(x))
        x = self.loss(self.linear2(x))
        x = self.linear3(x)
        return x #BASTARD 

