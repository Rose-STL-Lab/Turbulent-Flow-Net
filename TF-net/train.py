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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
import kornia
warnings.filterwarnings("ignore")

class Dataset(data.Dataset):
    def __init__(self, indices, input_length, mid, output_length, direc, stack_x):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.stack_x = stack_x
        self.direc = direc
        self.list_IDs = indices
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        if self.stack_x:
            x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid].reshape(-1, y.shape[-2], y.shape[-`])
        else:
            x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid]
        y = torch.load(self.direc + str(ID) + ".pt")[self.mid:(self.mid+self.output_length)]
        return x.float(), y.float()
    
def train_epoch(train_loader, model, optimizer, loss_function, coef = 0, regularizer = None):
    train_mse = []
    for xx, yy in train_loader:
        loss = 0
        ims = []
        xx = xx.to(device)
        yy = yy.to(device)
    
        for y in yy.transpose(0,1):
            im = model(xx)
            xx = torch.cat([xx[:, 2:], im], 1)
      
            if coef != 0 :
                loss += loss_function(im, y) + coef*regularizer(im, y)
            else:
                loss += loss_function(im, y)
            ims.append(im.cpu().data.numpy())
            
        ims = np.concatenate(ims, axis = 1)
        train_mse.append(loss.item()/yy.shape[1]) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    return train_mse

def eval_epoch(valid_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy in valid_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            ims = []


            for y in yy.transpose(0,1):
                im = model(xx)
                xx = torch.cat([xx[:, 2:], im], 1)
                loss += loss_function(im, y)
                ims.append(im.cpu().data.numpy())
  
            ims = np.array(ims).transpose(1,0,2,3,4)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())
            valid_mse.append(loss.item()/yy.shape[1])
        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)  
        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_mse, preds, trues

def test_epoch(test_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        loss_curve = []
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            
            loss = 0
            ims = []

            for y in yy.transpose(0,1):
                im = model(xx)
                xx = torch.cat([xx[:, 2:], im], 1)
                mse = loss_function(im, y)
                loss += mse
                loss_curve.append(mse.item())
                
                ims.append(im.cpu().data.numpy())

            ims = np.array(ims).transpose(1,0,2,3,4)    
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())            
            valid_mse.append(loss.item()/yy.shape[1])

        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)
        
        valid_mse = round(np.mean(valid_mse), 5)
        loss_curve = np.array(loss_curve).reshape(-1,60)
        loss_curve = np.sqrt(np.mean(loss_curve, axis = 0))
    return preds, trues, loss_curve
