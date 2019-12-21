from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import losses 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")

class Dataset(data.Dataset):
    def __init__(self, beg, end, input_length, mid, output_length, direc, stack_x):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.stack_x = stack_x
        self.direc = direc
        self.list_IDs = list(range(beg,end))
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        if self.stack_x:
            x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid].reshape(-1,64, 64)
        else:
            x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid]
        y = torch.load(self.direc + str(ID) + ".pt")[self.mid:(self.mid+self.output_length)]
        return x.float(), y.float()
    
def train_epoch(train_loader, model_x, model_y, warp_x, warp_y,  optimizer_x, optimizer_y, div_coef, loss_functions):

    photo_loss = loss_functions[0]
    div_loss = loss_functions[1]
    
    train_mse = []
    
    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        loss = 0
        pl, dl = 0, 0
        ims = []

        for y in yy.transpose(0,1):

            w_x = model_x(xx) 
            im_x = warp_x(xx[:, -2].unsqueeze(1), w_x)
            
            w_y = model_y(xx)
            im_y = warp_y(xx[:, -1].unsqueeze(1), w_y)
            #print(im_y.shape)
            im = torch.cat([im_x, im_y], dim = 1)
            #print(im.shape)
            xx = torch.cat([xx[:, 2:], im], 1)
            #print(im.shape)
            #print(y.unsqueeze(1).shape)
            pl += photo_loss(im, y)
            #print(pl.item())
            dl += div_loss(im)
            

            ims.append(im.cpu().data.numpy())
            
        ims = np.concatenate(ims, axis = 1)
        train_mse.append(pl.item()/yy.shape[1])
        loss = pl + div_coef * dl 
        
        optimizer_x.zero_grad()
        optimizer_y.zero_grad()
       
    
        loss.backward()
        
        optimizer_x.step()
        optimizer_y.step()
        
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    return train_mse

def eval_epoch(valid_loader, model_x, model_y, warp_x, warp_y, loss_fun):
    
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy in valid_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            loss = 0
            pl = 0
            ims = []
            
            for y in yy.transpose(0,1):

                w_x = model_x(xx)
                im_x = warp_x(xx[:, -2].unsqueeze(1), w_x)
                w_y = model_y(xx)
                im_y = warp_y(xx[:, -1].unsqueeze(1), w_y)
               # print(im_y.shape)
                im = torch.cat([im_x, im_y], dim = 1)
                #print(im.shape)
                xx = torch.cat([xx[:, 2:], im], 1)

                pl += loss_fun(im, y)
                ims.append(im.cpu().data.numpy())
            
            ims = np.array(ims).transpose(1,0,2,3,4)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())
            
            valid_mse.append(pl.item()/yy.shape[1])
        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)  
        valid_mse = round(np.sqrt(np.mean(valid_mse)),5)
    return valid_mse, preds, trues

def test_epoch(test_loader, model_x, model_y, warp_x, warp_y, loss_fun):
    
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        loss_curve = []
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            loss = 0
            pl = 0
            ims = []
            
            for y in yy.transpose(0,1):

                w_x = model_x(xx)
                im_x = warp_x(xx[:, -2].unsqueeze(1), w_x)
                w_y = model_y(xx)
                im_y = warp_y(xx[:, -1].unsqueeze(1), w_y)
               # print(im_y.shape)
                im = torch.cat([im_x, im_y], dim = 1)
                #print(im.shape)
                xx = torch.cat([xx[:, 2:], im], 1)
                mse = loss_fun(im, y)
                pl += mse
                loss_curve.append(mse.item())
                
                ims.append(im.cpu().data.numpy())
            
            ims = np.array(ims).transpose(1,0,2,3,4)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())
            
            valid_mse.append(pl.item()/yy.shape[1])
            
        loss_curve = np.array(loss_curve).reshape(-1,60)
        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)  
        valid_mse = round(np.mean(valid_mse), 5)
        loss_curve = np.sqrt(np.mean(loss_curve, axis = 0))
    return valid_mse, preds, trues, loss_curve

