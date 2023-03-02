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
from math import ceil
import random
import time
from model_addon import LES
from torch.autograd import Variable
from penalty import DivergenceLoss
from train import Dataset, train_epoch, eval_epoch, test_epoch, preprocess, Scaler
import warnings
warnings.filterwarnings("ignore")
import argparse
from icecream import ic as ic_print
from args import parse_arguments
import aim

# Reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
                        


args = parse_arguments()
args.use_test_mode = not args.not_use_test_mode

run = aim.Run(experiment=args.data)
run['args'] = vars(args)
run.description = args.desc

torch.manual_seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
ic_print(args.seed)


if args.data == "rbc_data.pt":
    compress = True
    permute = False
    offset=0
    transform_type = 'std'
    test_mode_train = False
elif args.data == 'data5.pt':
    compress = False
    permute = True
    offset = 60
    transform_type = 'std'
    split_spatially = False
    test_mode_train = True
elif 'data' in args.data and '101' in args.data:
    compress = False
    permute = True
    offset = 60
    transform_type = 'std'
    test_mode_train = True
else:
    raise ValueError("Un expected data file name")
args.transform = Scaler(transform_type, offset)
run['transform'] = vars(args.transform)

data_prep = preprocess(args, permute, compress, test_mode_train)
device_ids = args.d_ids
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

print("run lya model using device:",device)
#best_params: kernel_size 3, learning_rate 0.001, dropout_rate 0, batch_size 120, input_length 26, output_length 4
min_mse = args.min_mse
time_range  = args.time_range
output_length = args.outln_init
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
            dropout_rate = dropout_rate, time_range = time_range, addon_enc=args.addon_enc, addon_dec=args.addon_dec).to(device)
model = nn.DataParallel(model, device_ids=device_ids)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("num_params: ", num_params)

assert args.output_length >= args.outln_init
if args.outln_stride is None:
    args.outln_stride = args.output_length - args.outln_init
if args.outln_steps != 0:
    outln_rate = ceil((args.output_length - args.outln_init) / args.outln_stride) / args.outln_steps    
else:
    outln_rate = 0
    args.outln_init = args.output_length
    
valid_set = Dataset(valid_indices, input_length + time_range - 1, 40, 6, data_prep, stack_x=True, test_mode_train=test_mode_train)
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
            m_pred.bias = nn.Parameter(-args.m_init*torch.ones_like(m_pred.bias))
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
                            ], learning_rate, betas = (0.9, 0.999), weight_decay = args.wt_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = 0.9)

train_mse = []
train_reg = []
valid_mse = []
val_reg = []
test_mse = []

for i in range(args.epoch):
    start = time.time()
    scheduler.step()

    if i <= args.outln_steps:    
        output_length = min(int(outln_rate * i)*args.outln_stride + args.outln_init , args.output_length)
        train_set = Dataset(train_indices, input_length + time_range - 1, 40, output_length, data_prep, stack_x=True, test_mode_train=test_mode_train, noise=args.noise, do_not_scale_noise = args.dnsn)
        train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)

    ic_print(output_length)
    model.train()
    train_mse_rst, train_reg_rst = train_epoch(args, train_loader, model, optimizer, loss_fun, m_pred, coef, regularizer,coef2,cur_epoch=i,barrier=args.barrier,mide=args.mide,slope=args.slope,device=device)
    train_mse.append(train_mse_rst)
    train_reg.append(train_reg_rst)
    model.eval()
    mse, val_reg_rst,preds, trues = eval_epoch(valid_loader, model, loss_fun,coef2,barrier=args.barrier,mide=args.mide,slope=args.slope, device=device)
    valid_mse.append(mse)
    val_reg.append(val_reg_rst)
    run.track({'train_mse': train_mse_rst, 'train_reg': train_reg_rst}, context={'subset': 'train'}, epoch=i)
    run.track({'val_mse': mse, 'val_reg': val_reg_rst}, context={'subset': 'val'}, epoch=i)
    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1]   
        torch.save(model, args.path+"model.pth")
        torch.save(model.module.state_dict(), args.path+"module_stdict.pth")
        run.track(min_mse, name='min_valid_mse', epoch=i)
    end = time.time()
    # change 50 to 100
    #if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
    #        break
    ic_print(i, train_mse[-1],train_reg[-1], valid_mse[-1],val_reg[-1], round((end-start)/60,5))
    if m_pred is not None:
        ic_print(m_pred.m_pred.weight, m_pred.m_pred.bias, m_pred.slope)
ic_print(time_range, min_mse)

batch_size=21
if len(args.d_ids) >= 7:
    device_ids = args.d_ids[:7]
    batch_size = 28
elif len(args.d_ids) >= 4:
    device_ids = args.d_ids[:3]
    batch_size = 12

# model = LES(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size, 
#             dropout_rate = dropout_rate, time_range = time_range, addon_enc=args.addon_enc, addon_dec=args.addon_dec).to(device)
# model = nn.DataParallel(model, device_ids=device_ids)
# model.load_state_dict(torch.load(args.path+"model.pth"))
# torch.save(model, args.path+"model.pth")
# torch.save(model.module.state_dict(), args.path+"module_stdict.pth")

loss_fun = torch.nn.MSELoss()
best_model = nn.DataParallel(torch.load(args.path+"model.pth", map_location=device).module, device_ids=device_ids)
data_prep = preprocess(args, permute, compress, test_mode=args.use_test_mode)

# on val set
print("Validation in test setting")
test_set = Dataset(valid_indices, input_length + time_range - 1, 40, 60, data_prep, stack_x=True, test_mode=args.use_test_mode, test_mode_train=test_mode_train)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
preds, trues, loss_curve = test_epoch(args, test_loader, best_model, loss_fun,test_mode=not test_mode_train and args.use_test_mode,device=device)

torch.save({"loss_curve": loss_curve}, 
            args.path+f"results_val{'' if args.use_test_mode else '_64'}.pt",pickle_protocol=5)
for i, lc in enumerate(loss_curve):
    run.track(args.transform.beta.numpy() * lc, name=f"val_test_curve{'' if args.use_test_mode else '_64'}", context={'subset': 'test'}, step=i)

# On test set
if not args.only_val:
    print("Testing in test setting")
    test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, data_prep, stack_x=True, test_mode=args.use_test_mode, test_mode_train=test_mode_train)
    test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
    preds, trues, loss_curve = test_epoch(args, test_loader, best_model, loss_fun,test_mode=not test_mode_train and args.use_test_mode,device=device)

    torch.save({"loss_curve": loss_curve}, 
                args.path+f"results{'' if args.use_test_mode else '_64'}.pt",pickle_protocol=5)
    for i, lc in enumerate(loss_curve):
        run.track(args.transform.beta.numpy() * lc, name=f"test_curve{'' if args.use_test_mode else '_64'}", context={'subset': 'test'}, step=i)
