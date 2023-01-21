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
import warnings
warnings.filterwarnings("ignore")
import argparse
from icecream import ic as ic_print

# Reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_mse",
                        type=int,
                        default=1)
    parser.add_argument("--seed",
                        type=int,
                        default=53)
    parser.add_argument("--time_range",
                        type=int,
                        default=6)
    parser.add_argument("--output_length",
                        type=int,
                        default=4)
    parser.add_argument("--input_length",
                        type=int,
                        default=26)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001)
    parser.add_argument("--dropout_rate",
                        type=int,
                        default=0)
    parser.add_argument("--kernel_size",
                        type=int,
                        default=3)
    parser.add_argument("--batch_size",
                        type=int,
                        default=32)
    parser.add_argument("--coef",
                        type=float,
                        default=0)
    parser.add_argument("--coef2",
                        type=float,
                        default=1)
    parser.add_argument("--step_size",
                        type=int,
                        default=1)
    parser.add_argument("--path",
                        type=str,
                        default="./")
    parser.add_argument("--epoch",
                        type=int,
                        default=100)
    parser.add_argument("--mide",
                        type=float,
                        default=None)
    parser.add_argument("--slope",
                        type=int,
                        default=None) # 300
    parser.add_argument("--slope_init",
                        help="if slope is None, i.e. slope is learnt, init it with this value",
                        type=int,
                        default=50) # 300
    parser.add_argument("--barrier",
                        type=float,
                        default=1e-3)
    parser.add_argument("--d_ids",
                        nargs='+',
                        type=int,
                        default=[2])
    parser.add_argument("--bnorm", action='store_true', default=False)
    parser.add_argument("--no_weight", action='store_true', default=False)
    parser.add_argument("--only_val", action='store_true', default=False)
    parser.add_argument("--use_time", help="use time as well for (mse - mide) transformation", action='store_true', default=False)
    parser.add_argument("--time_factor",
                        type=float,
                        default=0.01)
    parser.add_argument("--desc",
                        type=str,
                        default="")
    return parser.parse_args()
                        


def preprocess(test_mode=False):
    data = torch.load("rbc_data.pt")

    # standardization
    std = torch.std(data)
    avg = torch.mean(data)
    print(std,avg)
    data = (data - avg)/std
    data = data[:,:,::4,::4]

    # divide each rectangular snapshot into 7 subregions
    # data_prep shape: num_subregions * time * channels * w * h
    if not test_mode:
        data_prep = torch.FloatTensor(torch.stack([data[:,:,:,k*64:(k+1)*64] for k in range(7)]))
        #print(data_prep.shape)
    else:
        data_prep = torch.FloatTensor(data) # full domain
    return data_prep

args = parse_arguments()

torch.manual_seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
ic_print(args.seed)

data_prep = preprocess()
device_ids = args.d_ids
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

print("run lya model using device:",device)
#best_params: kernel_size 3, learning_rate 0.001, dropout_rate 0, batch_size 120, input_length 25, output_length 4
min_mse = args.min_mse
time_range  = args.time_range
output_length = args.output_length
input_length = args.input_length
learning_rate = args.learning_rate
dropout_rate = args.dropout_rate
kernel_size = args.kernel_size
batch_size = args.batch_size
step_size = args.step_size

train_indices = list(range(0, 6000))
valid_indices = list(range(6000, 7700))
test_indices = list(range(7700, 9800))

model = LES(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size, 
            dropout_rate = dropout_rate, time_range = time_range).to(device)
model = nn.DataParallel(model, device_ids=device_ids)

train_set = Dataset(train_indices, input_length + time_range - 1, 40, output_length, data_prep, True)
valid_set = Dataset(valid_indices, input_length + time_range - 1, 40, 6, data_prep, True)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)
loss_fun = torch.nn.MSELoss()
regularizer = DivergenceLoss(torch.nn.MSELoss()) #Has cuda leak to zeroth device
coef = args.coef
coef2 = args.coef2
print("configs:",[min_mse,time_range,output_length,input_length,learning_rate,dropout_rate,kernel_size,batch_size,coef,coef2,step_size,args])

if args.mide is None or args.slope is None:
    assert args.mide is None
    class Mide_pred(nn.Module):
        def __init__(self, ):
            super(Mide_pred, self).__init__()
            m_pred = nn.Linear(2 if args.use_time else 1,1)
            m_pred.weight = nn.Parameter(torch.ones_like(m_pred.weight))
            if args.use_time:
                m_pred.weight.data[:,1] = 0.0
            m_pred.bias = nn.Parameter(-0.09*torch.ones_like(m_pred.bias))
            self.m_pred = m_pred

            if args.slope is None:
                self.slope = nn.Parameter(args.slope_init*torch.ones(1))
            else:
                self.slope = None

        def forward(self, x):
            return self.m_pred(x)

    m_pred = Mide_pred().to(device)
else:
    m_pred = None

optimizer = torch.optim.Adam([
                                {'params': model.parameters()},
                                {'params': ([] if m_pred is None else m_pred.parameters()), 'lr':1e-4, 'weight_decay':0.0}
                            ], learning_rate, betas = (0.9, 0.999), weight_decay = 4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = 0.9)

train_mse = []
train_reg = []
valid_mse = []
val_reg = []
test_mse = []

for i in range(args.epoch):
    start = time.time()
    scheduler.step()
    model.train()
    train_mse_rst,train_reg_rst = train_epoch(args, train_loader, model, optimizer, loss_fun, m_pred, coef, regularizer,coef2,cur_epoch=i,barrier=args.barrier,mide=args.mide,slope=args.slope,device=device)
    train_mse.append(train_mse_rst)
    train_reg.append(train_reg_rst)
    model.eval()
    mse, val_reg_rst,preds, trues = eval_epoch(valid_loader, model, loss_fun,coef2,barrier=args.barrier,mide=args.mide,slope=args.slope, device=device)
    valid_mse.append(mse)
    val_reg.append(val_reg_rst)
    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1]   
        torch.save(model, args.path+"model.pth")
    end = time.time()
    # change 50 to 100
    #if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
    #        break
    ic_print(i, train_mse[-1],train_reg[-1], valid_mse[-1],val_reg[-1], round((end-start)/60,5))
    ic_print(m_pred.m_pred.weight, m_pred.m_pred.bias, m_pred.slope)
ic_print(time_range, min_mse)

batch_size=21
if len(args.d_ids) >= 7:
    device_ids = args.d_ids[:7]
    batch_size = 28
elif len(args.d_ids) >= 4:
    device_ids = args.d_ids[:3]
    batch_size = 12

loss_fun = torch.nn.MSELoss()
best_model = nn.DataParallel(torch.load(args.path+"model.pth", map_location=device).module, device_ids=device_ids)
data_prep = preprocess(test_mode=True)

# on val set
print("Validation in test setting")
test_set = Dataset(valid_indices, input_length + time_range - 1, 40, 60, data_prep, True,test_mode=True)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun,test_mode=True,device=device)

torch.save({"loss_curve": loss_curve}, 
            args.path+"results_val.pt",pickle_protocol=5)

# On test set
if not args.only_val:
    print("Testing in test setting")
    test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, data_prep, True,test_mode=True)
    test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
    preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun,test_mode=True,device=device)

    torch.save({"loss_curve": loss_curve}, 
                args.path+"results.pt",pickle_protocol=5)
