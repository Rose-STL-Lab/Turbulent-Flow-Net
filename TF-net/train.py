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
import warnings
import kornia
from tqdm import tqdm
warnings.filterwarnings("ignore")

class Dataset(data.Dataset):
    def __init__(self, indices, input_length, mid, output_length, data_prep, stack_x,test_mode=False): # test_mode: full areas or not
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.stack_x = stack_x
        self.data_prep = data_prep
        self.list_IDs = indices
        self.test_mode = test_mode
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        j = ID // 7
        #i = ID % 7
        if self.test_mode:
            data_ = self.data_prep[j:j+100]#[i,j:j+100]
        else:
            i = ID % 7
            data_ = self.data_prep[i,j:j+100]
        y = data_[self.mid:(self.mid+self.output_length)]
        if self.stack_x:
            x = data_[(self.mid-self.input_length):self.mid].reshape(-1, y.shape[-2], y.shape[-1])
        else:
            x = data_[(self.mid-self.input_length):self.mid]
        
        return x.float(), y.float()
    
def train_epoch(args, train_loader, model, optimizer, loss_function, m_pred= None, coef = 0, regularizer = None, coef2=1.0,cur_epoch=-1,barrier=1e2,mide=None,slope=None, 
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if slope is None:
        slope = m_pred.slope
    train_mse = []
    train_reg = []
    training_data = tqdm(train_loader)

    for _,(xx, yy) in enumerate(training_data):
        loss = 0
        batch_reg = 0
        ims = []
        xx = xx.to(device) # batch*time*width*height
        yy = yy.to(device).transpose(0,1)
        
        length = len(yy)
        
        prev_lya = None # for approximating dV/dt~V(t+1)-V(t)
        pred_losses = []
    
        for cur_t, y in enumerate(yy):
            #print("xx:",xx.shape,"yy:",yy.shape,y.shape)
            im = model(xx)
            xx = torch.cat([xx[:, 2:], im], 1)
            
            pred_loss = loss_function(im, y)
            lya_val = lyapunov_func(im,y)  # (Batch, )
            
            if coef != 0:
                loss += pred_loss + coef*regularizer(im, y)
            else:
                loss += pred_loss

            pred_losses.append(pred_loss.item())
                
            if coef2 != 0 and prev_lya != None:
                # dV_dt = torch.nn.functional.relu(lya_val - prev_lya)
                #dV_dt = F.relu(lya_val - prev_lya)
                #print(dV_dt.detach().cpu().numpy())
                #lya_reg = torch.mean(dV_dt) # = dV_dt
                # loss += coef2*lya_reg
                dV_dt = lya_val - prev_lya # (Batch, )
                if m_pred is not None:
                    temp = lya_val.reshape((-1,1))
                    if args.use_time:
                        temp = torch.hstack((temp, cur_t*args.time_factor*torch.ones_like(temp)))
                    cur_pred_error = m_pred(temp).reshape((-1,))   # Send mide as zero, and update cur_pred_error
                    lya_reg,log_c,relu_c,log_v,relu_v = log_barrier(args,dV_dt,coef2,barrier,0.0,slope, cur_pred_error)
                else:
                    lya_reg,log_c,relu_c,log_v,relu_v = log_barrier(args,dV_dt,coef2,barrier,mide,slope, lya_val)
                if lya_reg != None:
                    loss += lya_reg
                    batch_reg += lya_reg.item()
                
            ims.append(im.cpu().data.numpy())
            
            prev_lya = lya_val
            
        ims = np.concatenate(ims, axis = 1)
        train_mse.append(loss.item()/length) 
        train_reg.append(batch_reg/length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if coef2 != 0:
            training_data.set_postfix(cur_epoch=cur_epoch,log_p=log_c/(log_c+relu_c),log_v=log_v,relu_v=relu_v, bias_data=m_pred.m_pred.bias.data)
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    train_reg = round(np.mean(train_reg),5)
    return train_mse,train_reg

def eval_epoch(valid_loader, model, loss_function,coef2=1.0,barrier=1e2,mide=None,slope=None, device=None):
    valid_mse = []
    val_reg = []
    preds = []
    trues = []
    with torch.no_grad():
        log_c_t,relu_c_t = 0,0
        for xx, yy in valid_loader:
            loss = 0
            batch_reg = 0
            xx = xx.to(device)
            yy = yy.to(device).transpose(0,1)
            ims = []
            
            length = len(yy)
            
            prev_lya = None # for approximating dV/dt~V(t+1)-V(t)
            
            for y in yy:
                im = model(xx)
                xx = torch.cat([xx[:, 2:], im], 1)
                pred_loss = loss_function(im,y)
                loss += pred_loss
                lya_val = lyapunov_func(im,y)
                #if coef2 != 0 and prev_lya != None:
                #    # dV_dt = torch.nn.functional.relu(lya_val - prev_lya)
                #    #dV_dt = F.relu(lya_val - prev_lya)
                #    #lya_reg = torch.mean(dV_dt) # = dV_dt
                #    #loss += coef2*lya_reg
                #    dV_dt = lya_val - prev_lya
                #    lya_reg,log_c,relu_c,_,_ = log_barrier(dV_dt,coef2,barrier,mide,slope,lya_val)
                #    log_c_t += log_c
                #    relu_c_t += relu_c
                #    loss += lya_reg
                #    batch_reg += lya_reg.item()
                # ims.append(im.cpu().data.numpy())
                prev_lya = lya_val
  
            # ims = np.array(ims).transpose(1,0,2,3,4)
            # preds.append(ims)
            # trues.append(yy.transpose(0,1).cpu().data.numpy())
            valid_mse.append(loss.item()/length)
            val_reg.append(batch_reg/length)
        # preds = np.concatenate(preds, axis = 0)  
        # trues = np.concatenate(trues, axis = 0)  
        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
        val_reg = round(np.mean(val_reg),5)
        print("log_c_t:",log_c_t,"relu_c_t:",relu_c_t)
    return valid_mse, val_reg,preds, trues

def test_epoch(test_loader, model, loss_function,test_mode=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    valid_mse = []
    # preds = []
    # trues = []
    preds = trues = None
    with torch.no_grad():
        loss_curve = []
        test_data = tqdm(test_loader)
        for _,(xx, yy) in enumerate(test_data):
        # for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            
            loss = 0
            ims = []

            for y in yy.transpose(0,1):
                try:
                    im = model(xx,test_mode=test_mode)
                except TypeError as err:
                    tqdm.write(f"{xx.shape}")
                    raise TypeError(err)
                xx = torch.cat([xx[:, 2:], im], 1)
                mse = loss_function(im, y)
                loss += mse
                loss_curve.append(mse.item())
                
                ims.append(im.cpu().data.numpy())

            ims = np.array(ims).transpose(1,0,2,3,4)    
            # preds.append(ims)
            # trues.append(yy.cpu().data.numpy())            
            valid_mse.append(loss.item()/yy.shape[1])

        # preds = np.concatenate(preds, axis = 0)  
        # trues = np.concatenate(trues, axis = 0)
        
        valid_mse = round(np.mean(valid_mse), 5)
        loss_curve = np.array(loss_curve).reshape(-1,60)
        loss_curve = np.sqrt(np.mean(loss_curve, axis = 0))
        print(loss_curve*4506.0068)
    return preds, trues, loss_curve
    
def lyapunov_func(im,y,f=F.mse_loss): # batch*tensor --> R
    # MSE
    #mse = f(im,y)
    dims = tuple(range(1,len(im.shape)))
    mse = torch.mean((im-y)**2,dim=dims)    # y doesn't have t dimension
    return mse

def log_barrier(args, dV_dt,coef2,t=1,mide=None,slope=None,cur_pred_mse=None): # remove relu in dV_dt
    log_sum = 0
    relu_sum = 0
    log_count = 0
    relu_count = 0
    dV_dt = dV_dt
    if not args.no_weight and slope is not None:
        assert dV_dt.shape[0] == cur_pred_mse.shape[0], "batch not match"
        weights = coef2 / (1+torch.e**(-slope*(cur_pred_mse-mide)))
    else:
        weights = coef2 * np.array([1 for i in range(len(dV_dt))])
    idx = 0
    for h in dV_dt:
        h_ = h.detach().cpu().item()
        if h_ < 0:
            log_sum += -weights[idx] * torch.log(-h)/t
            log_count += 1
        elif h_ >= 0:
            relu_sum += weights[idx] * h
            relu_count += 1
        idx += 1
    #return (log_sum + coef2 * relu_sum) / (log_count + relu_count),log_count,relu_count,\
    #return (log_sum) / (log_count) if log_count != 0 else None,log_count,relu_count,\
    #    log_sum.detach().cpu().item() if type(log_sum) != int else 0,relu_sum.detach().cpu().item() if type(relu_sum) != int else 0
    return (relu_sum) / (relu_count + (log_count if args.bnorm else 0)) if type(relu_sum) != int else None,log_count,relu_count,\
        log_sum.detach().cpu().item() if type(log_sum) != int else 0,relu_sum.detach().cpu().item() if type(relu_sum) != int else 0
