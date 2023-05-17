import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(data.Dataset):
    def __init__(self, indices, input_length, mid, output_length, direc, stack_x):
        # Constructor for the Dataset class
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.stack_x = stack_x
        self.direc = direc
        self.list_IDs = indices
        
    def __len__(self):
        # Returns the length of the dataset
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Returns a specific item from the dataset given an index
        ID = self.list_IDs[index]
        y = torch.load(self.direc + str(ID) + ".pt")[self.mid:(self.mid+self.output_length)]
        
        if self.stack_x:  # Stack input along the channel dimension
            x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid].reshape(-1, y.shape[-2], y.shape[-1]) 
        else:
            x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid]
            
        return x.float(), y.float()

def train_epoch(train_loader, model, optimizer, loss_function, coef=0, regularizer=None):
    # Train the model for one epoch
    train_mse = []
    for xx, yy in train_loader:
        loss = 0
        ims = []
        xx = xx.float().to(device)
        yy = yy.float().to(device)
        # print(xx.shape)
    
        for y in yy.transpose(0, 1):
            im = model(xx)
            xx = torch.cat([xx[:, im.shape[1]:], im], 1)
            if coef != 0:
                loss += loss_function(im, y) + coef * regularizer(im)
            else:
                loss += loss_function(im, y)
            ims.append(im.cpu().data.numpy())
            
        ims = np.concatenate(ims, axis=1)
        train_mse.append(loss.item() / yy.shape[1]) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_rmse = round(np.sqrt(np.mean(train_mse)), 5)
    return train_rmse

def eval_epoch(valid_loader, model, loss_function):
    # Evaluate the model on the validation set
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy in valid_loader:
            loss = 0
            xx = xx.float().to(device)
            yy = yy.float().to(device)
            ims = []

            for y in yy.transpose(0, 1):
                im = model(xx)
                xx = torch.cat([xx[:, im.shape[1]:], im], 1)
                loss += loss_function(im, y)
                ims.append(im.cpu().data.numpy())
  
            ims = np.array(ims).transpose(1, 0, 2, 3, 4)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())
            valid_mse.append(loss.item() / yy.shape[1])
        preds = np.concatenate(preds, axis=0)  
        trues = np.concatenate(trues, axis=0)  
        valid_rmse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_rmse, preds, trues


def test_epoch(test_loader, model, loss_function):
    # Test the model on the test set
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        mse_curve = []
        for xx, yy in test_loader:
            xx = xx.float().to(device)
            yy = yy.float().to(device)
            
            loss = 0
            ims = []

            for y in yy.transpose(0, 1):
                im = model(xx)
                xx = torch.cat([xx[:, im.shape[1]:], im], 1)
                mse = loss_function(im, y)
                loss += mse
                mse_curve.append(mse.item())
                
                ims.append(im.cpu().data.numpy())

            ims = np.array(ims).transpose(1, 0, 2, 3, 4)    
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())            
            valid_mse.append(loss.item() / yy.shape[1])

        preds = np.concatenate(preds, axis=0)  
        trues = np.concatenate(trues, axis=0)
        
        valid_rmse = round(np.mean(valid_mse), 5)
        mse_curve = np.array(mse_curve).reshape(-1, 60)
        rmse_curve = np.sqrt(np.mean(mse_curve, axis=0))
    return preds, trues, rmse_curve
