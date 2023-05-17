###### ResNet ######
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################ResNett################################
# 18-layer ResNet
class Resblock(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super(Resblock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU()
        ) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU()
        ) 
        
        if input_channels != hidden_dim:
            self.upscale = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2),
                nn.LeakyReLU()
                )        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        
    def forward(self, xx):
        out = self.layer1(xx)  
        if self.input_channels != self.hidden_dim:
            out = self.layer2(out) + self.upscale(xx)
        else:
            out = self.layer2(out) + xx
        return out
    

class ResNet(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(ResNet, self).__init__()
        layers = [Resblock(input_channels, 64, kernel_size), Resblock(64, 64, kernel_size), Resblock(64, 64, kernel_size)]
        layers += [Resblock(64, 128, kernel_size), 
                   Resblock(128, 128, kernel_size),
                   Resblock(128, 128, kernel_size), 
                   Resblock(128, 128, kernel_size)]
        layers += [Resblock(128, 256, kernel_size),
                   Resblock(256, 256, kernel_size), 
                   Resblock(256, 256, kernel_size),
                   Resblock(256, 256, kernel_size)]
        layers += [Resblock(256, 512, kernel_size), 
                   Resblock(512, 512, kernel_size), 
                   Resblock(512, 512, kernel_size)]
        layers += [nn.Conv2d(512, output_channels, kernel_size = kernel_size, padding = (kernel_size-1)//2)]
        self.model = nn.Sequential(*layers)
             
    def forward(self, xx):
        out = self.model(xx)
        return out
