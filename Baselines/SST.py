from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data
import itertools
import re
import random
import time
import losses as losses
from torch.autograd import Variable
from train import train_epoch, eval_epoch, test_epoch, Dataset
from warp import GaussianWarpingScheme
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")
def conv(in_planes, output_channels, kernel_size, stride, dropout_rate):
    return nn.Sequential(
        nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size - 1) // 2, bias = False),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_rate)
    )

def deconv(input_channels, output_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                           stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True)
    )

def output_layer(input_channels, output_channels, kernel_size, stride, dropout_rate):
    return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                     stride=stride, padding=(kernel_size - 1) // 2)

class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = conv(input_channels, 64, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2 = conv(64, 128, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3 = conv(128, 256, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = conv(256, 256, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv4 = conv(256, 512, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv4_1 = conv(512, 512, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv5 = conv(512, 1024, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv5_1 = conv(1024, 1024, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)

        self.deconv4 = deconv(1024, 256)
        self.deconv3 = deconv(768, 128)
        self.deconv2 = deconv(384, 64)
        self.deconv1 = deconv(192, 32)
        self.deconv0 = deconv(96, 16)
    
        self.output_layer = output_layer(16 + input_channels, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.002/n)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))

        out_deconv4 = self.deconv4(out_conv5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out
    
class GaussianWarpingScheme(nn.Module):
    def __init__(self, padding_mode='zeros', std=0.25):
        super(GaussianWarpingScheme, self).__init__()
        self.grid = DenseGridGen()
        self.std = std
        self.padding_mode = padding_mode

    def forward(self, im, w):
        return F.grid_sample(im, self.grid(w), padding_mode=self.padding_mode, mode='bilinear')
    
train_direc = "/global/cscratch1/sd/rwang2/Data/data_64/train_seqs/train_sample_"
test_direc = "/global/cscratch1/sd/rwang2/Data/data_64/test_seqs/test_sample_"
input_length = 24
min_mse = 1
batch_size = 50
output_length = 6
div_coef = 5

epochs = 100
learning_rate = 0.001
train_set = Dataset(0, 6000, input_length, 30, output_length, train_direc, True)
valid_set = Dataset(0, 2170, input_length, 30, 6, test_direc, True)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)


model_x = nn.DataParallel(U_net(input_channels = input_length*2, output_channels = 2, 
                                kernel_size = 3, dropout_rate = 0).to(device))
model_y = nn.DataParallel(U_net(input_channels = input_length*2, output_channels = 2, 
                                kernel_size = 3, dropout_rate = 0).to(device))

optimizer_x = torch.optim.Adam(model_x.parameters(), learning_rate,
                     betas=(0.9, 0.999), weight_decay=4e-4)

optimizer_y = torch.optim.Adam(model_y.parameters(), learning_rate,
                     betas=(0.9, 0.999), weight_decay=4e-4)

scheduler_x = torch.optim.lr_scheduler.StepLR(optimizer_x, step_size= 5, gamma=0.8)
scheduler_y = torch.optim.lr_scheduler.StepLR(optimizer_y, step_size= 5, gamma=0.8)


warp_x = GaussianWarpingScheme().to(device)
warp_y = GaussianWarpingScheme().to(device)

photo_loss = torch.nn.MSELoss()
div_loss = losses.DivergenceLoss(torch.nn.MSELoss())
loss_functions = [photo_loss, div_loss]
train_mse = []
valid_mse = []
test_mse = []

for i in range(100):
    start = time.time()
    scheduler_x.step()
    scheduler_y.step()

    model_x.train()
    warp_x.train()
    model_y.train()
    warp_y.train()


    train_mse.append(train_epoch(train_loader, model_x, model_y, warp_x, warp_y,  
                                 optimizer_x, optimizer_y, div_coef, loss_functions))
    model_x.eval()
    warp_x.eval()
    model_y.eval()
    warp_y.eval()

    mse, preds, trues = eval_epoch(valid_loader, model_x, model_y, warp_x, warp_y, photo_loss)
    valid_mse.append(mse)
    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = [model_x, warp_x, model_y, warp_y]
        best_preds = preds

    end = time.time()
    print(train_mse[-1], valid_mse[-1], round((end-start)/60,5))
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
            
print(input_length, min_mse)