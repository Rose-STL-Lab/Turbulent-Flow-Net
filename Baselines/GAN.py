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
from torch.autograd import Variable
from model import U_net
from train import Dataset, test_epoch
import os
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")
import kornia
import itertools as it
from random import sample 

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size - 1) // 2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, inplace=True),
    )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                           stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True),
    )

def predict_flow(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

class U_net(nn.Module):
    def __init__(self, input_channels=4, output_channels=1):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = conv(input_channels, 64, kernel_size=3, stride=2)
        self.conv2 = conv(64, 128, kernel_size=3, stride=2)
        self.conv3 = conv(128, 256, kernel_size=3, stride=2)
        self.conv3_1 = conv(256, 256, kernel_size=3)
        self.conv4 = conv(256, 512, kernel_size=3, stride=2)
        self.conv4_1 = conv(512, 512, kernel_size=3)
        self.conv5 = conv(512, 1024, kernel_size=3, stride=2)
        #self.conv5_1 = conv(1024, 1024)

        self.deconv4 = deconv(1024, 256)
        self.deconv3 = deconv(768, 128)
        self.deconv2 = deconv(384, 64)
        self.deconv1 = deconv(192, 32)
        self.deconv0 = deconv(96, 16)
    
        self.predict_flow0 = predict_flow(16 + input_channels, output_channels)

    def forward(self, x):

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5(out_conv4)

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
        flow0 = self.predict_flow0(concat0)

        return flow0
    
class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()
        self.model = U_net(input_channels = input_channels, output_channels = output_channels)
        
    def forward(self, xx, output_steps):
        ims = []
        for i in range(output_steps):
            im = self.model(xx)
            ims.append(im)
            xx = torch.cat([xx[:, 2:], im], 1)
        return torch.cat(ims, dim = 1)
    
class Discriminator_Spatial(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator_Spatial, self).__init__()
        self.activ = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size = 5, padding = 2, stride = 2),
            nn.BatchNorm2d(32)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 5, padding = 2, stride = 2),
            nn.BatchNorm2d(64)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 5, padding = 2, stride = 2),
            nn.BatchNorm2d(128)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(256)
        )
        
        self.dense_layer = nn.Sequential(
            nn.Linear(256*8*8, 1),
            nn.Sigmoid()
        )

    def forward(self, ims):
        ims1 = self.activ(self.conv1(ims))
        ims2 = self.activ(self.conv2(ims1))
        ims3 = self.activ(self.conv3(ims2))
        ims4 = self.activ(self.conv4(ims3))
        
        out = ims4.reshape(ims4.shape[0], -1)
        out = self.dense_layer(out)
        return out,  Variable(ims1, requires_grad=True), Variable(ims2, requires_grad=True), Variable(ims3, requires_grad=True), Variable(ims4, requires_grad=True)

    
def noise(bz, div):
    return torch.rand(bz,1)/div


batch_size = 64
input_length = 25
losses = []
train_direc = "/global/cscratch1/sd/rwang2/TF-net/Data/data_64/sample_"
test_direc = "/global/cscratch1/sd/rwang2/TF-net/Data/data_64/sample_"
min_mse = 1

coef = 1
lr_g = 0.01  
lr_ds = 0.003
output_length = 5

# Data Loader
train_indices = list(range(0, 6000))
valid_indices = list(range(6000, 7700))
test_indices = list(range(7700, 9800))

train_set = Dataset(train_indices, input_length, 30, output_length, train_direc, True)
valid_set = Dataset(valid_indices, input_length, 30, 6, test_direc, True)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 12)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 12)


Gen = Generator(input_length*2, 2).to(device)
Dis_s = Discriminator_Spatial(output_length*2).to(device)
Gen = nn.DataParallel(Gen)
Dis_s = nn.DataParallel(Dis_s)

optimizer_G = torch.optim.Adam(Gen.parameters(), lr = lr_g, betas=(0.9, 0.999), weight_decay=4e-4)
optimizer_Ds = torch.optim.Adam(Dis_s.parameters(), lr = lr_ds, betas=(0.9, 0.999), weight_decay=4e-4)

scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size= 1, gamma=0.9)
scheduler_Ds = torch.optim.lr_scheduler.StepLR(optimizer_Ds, step_size= 1, gamma=0.9)

loss_fun = torch.nn.BCELoss()
loss_mse = torch.nn.MSELoss()

train_mse = []
valid_mse = []

for epoch in range(1000):
    beg = time.time()
    scheduler_G.step()
    scheduler_Ds.step()

    mse = []
    for xx, real_imgs in train_loader:
        xx, real_imgs = xx.to(device), real_imgs.to(device)
        valid = Variable(torch.Tensor(xx.size(0), 1).fill_(1.0) - noise(xx.size(0),10),requires_grad=False).to(device)
        fake = Variable(torch.Tensor(xx.size(0), 1).fill_(0.0) + noise(xx.size(0), 10),requires_grad=False).to(device)

        optimizer_G.zero_grad()
        gen_imgs = Gen(xx, output_length)

        ## Generator
        g_loss1 = loss_fun(Dis_s(gen_imgs)[0], valid) 
        g_loss = g_loss1  + coef*loss_mse(gen_imgs, real_imgs) 
        #- 0.0001*(loss_mse(dy1, gy1) + loss_mse(dy2, gy2) + loss_mse(dy3, gy3) + loss_mse(dy4, gy4)) \

        g_loss.backward()
        optimizer_G.step()
        
        ## Discriminator 
        optimizer_Ds.zero_grad()
        d_real, dy1, dy2, dy3, dy4 = Dis_s(real_imgs)
        d_fake, gy1, gy2, gy3, gy4 = Dis_s(gen_imgs.detach())
        real_loss_s = loss_fun(d_real, valid)
        fake_loss_s = loss_fun(d_fake, fake)
        ds_loss = real_loss_s + fake_loss_s

        ds_loss.backward()
        optimizer_Ds.step()

        

        mse.append(loss_mse(gen_imgs, real_imgs).item())
        
        losses.append([round(g_loss1.item(),3), round(real_loss_s.item(), 3), round(fake_loss_s.item(), 3)])

    train_mse.append(round(np.sqrt(np.mean(mse)),5))

    mse = []
    for xx, real_imgs in valid_loader:
        xx, real_imgs = xx.to(device), real_imgs.to(device)
        mse.append(loss_mse(Gen(xx, 6), real_imgs).item())

    valid_mse.append(round(np.sqrt(np.mean(mse)),5))

    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1]
        torch.save(Gen, "Gen.pth")

    end = time.time()
    print(train_mse[-1], valid_mse[-1], round((end-beg)/60,3))
    if (len(train_mse) > 70 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
        break
        

loss_mse = torch.nn.MSELoss()
model = torch.load("Gen.pth")
test_set = Dataset(test_indices, input_length, 30, 60, test_direc, True)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 12)
test_rmse, preds, trues, loss_curve = test_epoch(test_loader, model, loss_mse)
torch.save({"preds": preds, 
            "trues": trues, 
            "loss_curve": loss_curve},
            "GAN.pt")
   
