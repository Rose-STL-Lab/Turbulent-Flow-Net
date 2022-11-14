import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

class Resblock(nn.Module):
    def __init__(self, input_channels, hidden_dim, dropout_rate, res = True):
        super(Resblock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        ) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        ) 
        self.res = res
        
    def forward(self, x):
        out = self.layer1(x)
        if self.res:
            out = self.layer2(out) + x
        else:
            out = self.layer2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels, output_channels, dropout_rate):
        super(ResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        layers = [Resblock(64, 64, dropout_rate) for i in range(3)]
        layers += [Resblock(64, 128, dropout_rate, False)] + [Resblock(128, 128, dropout_rate) for i in range(3)]
        layers += [Resblock(128, 256, dropout_rate, False)] + [Resblock(256, 256, dropout_rate) for i in range(5)]
        layers += [Resblock(256, 512, dropout_rate, False)] + [Resblock(512, 512, dropout_rate) for i in range(2)]
        self.middle_layer = nn.Sequential(*layers)
        self.output_layer = nn.Conv2d(512, output_channels, kernel_size = 3, padding = 1)
       
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
              
    def forward(self, x):
        out = self.input_layer(x)
        out = self.middle_layer(out)
        out = self.output_layer(out)
        return out
    
    

