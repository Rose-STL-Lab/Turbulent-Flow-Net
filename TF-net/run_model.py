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
from model import LES
from torch.autograd import Variable
from penalty import DivergenceLoss
from train import Dataset, train_epoch, eval_epoch, test_epoch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")


train_direc = "/global/cscratch1/sd/rwang2/TF-net/Data/data_64/sample_"
test_direc = "/global/cscratch1/sd/rwang2/TF-net/Data/data_64/sample_"

#best_params: kernel_size 3, learning_rate 0.001, dropout_rate 0, batch_size 120, input_length 25, output_length 4
min_mse = 1
time_range  = 6
output_length = 4
input_length = 26
learning_rate = 0.001
dropout_rate = 0
kernel_size = 3
batch_size = 32

train_indices = list(range(0, 6000))
valid_indices = list(range(6000, 7700))
test_indices = list(range(7700, 9800))

model = LES(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size, 
            dropout_rate = dropout_rate, time_range = time_range).to(device)
model = nn.DataParallel(model)

train_set = Dataset(valid_indices, input_length + time_range - 1, 40, output_length, train_direc, True)
valid_set = Dataset(valid_indices, input_length + time_range - 1, 40, 6, test_direc, True)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)
loss_fun = torch.nn.MSELoss()
regularizer = DivergenceLoss(torch.nn.MSELoss())
coef = 0

optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas = (0.9, 0.999), weight_decay = 4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)

train_mse = []
valid_mse = []
test_mse = []
for i in range(100):
    start = time.time()
    torch.cuda.empty_cache()
    scheduler.step()
    model.train()
    train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun, coef, regularizer))#
    model.eval()
    mse, preds, trues = eval_epoch(valid_loader, model, loss_fun)
    valid_mse.append(mse)
    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model 
        torch.save(best_model, "model.pth")
    end = time.time()
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
    print(train_mse[-1], valid_mse[-1], round((end-start)/60,5))
print(time_range, min_mse)


loss_fun = torch.nn.MSELoss()
best_model = torch.load("model.pth")
test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, test_direc, True)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun)

torch.save({"preds": preds,
            "trues": trues,
            "loss_curve": loss_curve}, 
            "results.pt")
