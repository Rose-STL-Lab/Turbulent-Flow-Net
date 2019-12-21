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
from CDNN import ConvDeconvEstimator 
from warp import GaussianWarpingScheme
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")

train_direc = "/global/cscratch1/sd/rwang2/Data/data_64/train_seqs/train_sample_"
test_direc = "/global/cscratch1/sd/rwang2/Data/data_64/test_seqs/test_sample_"
input_length = 24
min_mse = 100
batch_size = 160
output_length = 6
div_coef = 5

epochs = 100
learning_rate = 0.0001
train_set = Dataset(0, 6000, input_length, 30, output_length, train_direc, True)
valid_set = Dataset(0, 2170, input_length, 30, 6, test_direc, True)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)


model_x = nn.DataParallel(ConvDeconvEstimator(input_channels = input_length*2).to(device))
model_y = nn.DataParallel(ConvDeconvEstimator(input_channels = input_length*2).to(device))

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

for i in range(2):
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
    if (len(train_mse) > 30 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
print(input_length, min_mse)