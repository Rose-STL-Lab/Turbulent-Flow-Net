import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils import data
from torch import autograd 
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(data.Dataset):
    def __init__(self, points, beg = 0, end = 1, sample = False, prop = 0.1):
        if sample:
            lst = np.random.choice(np.array(range(beg, end)), int((end-beg)*prop), replace=False)
        
            self.X = points[0,lst,:-1]
            self.U = points[0,lst,-1:]
            self.V = points[1,lst,-1:]
        else:
            self.X = points[0,beg:end,:-1]
            self.U = points[0,beg:end,-1:]
            self.V = points[1,beg:end,-1:]
            
            
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index]
        u = self.U[index]
        v = self.V[index]
        return x.float(), u.float(), v.float()
    
def train_epoch(train_loader, train_loader2, model, optimizer, loss_fun):
    mses = []
    Pr = 0.71  

    for inputs, u_target, v_target in train_loader:
        inputs = inputs.to(device)
        u_target = u_target.to(device)
        v_target = v_target.to(device)
        Loss = 0
        u, v, u_t, v_t, u_x, u_xx, u_y, u_yy, v_x, v_xx, v_y, v_yy, f_u, f_v = model(inputs, False)
         
        momen_u = u_t + u*u_x + v*u_y - Pr*(u_xx + u_yy) + f_u
        momen_v = v_t + u*v_x + v*v_y - Pr*(v_xx + v_yy) + f_v
        con_u = u_x + u_y
        con_v = v_x + v_y
         
        velocity_mse = (loss_fun(u, u_target) + loss_fun(v, v_target))/2
        momen_mse = (loss_fun(momen_u, momen_u.detach() * 0) + loss_fun(momen_v, momen_v.detach() * 0))/2
        con_mse = (loss_fun(con_u, con_u.detach() * 0) + loss_fun(con_v, con_v.detach() * 0))/2
        Loss = velocity_mse+ momen_mse + con_mse
        mses.append(velocity_mse.item())
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        
    for inputs, _, _ in train_loader2:
        inputs = inputs.to(device)
        Loss = 0
        u, v, u_t, v_t, u_x, u_xx, u_y, u_yy, v_x, v_xx, v_y, v_yy, f_u, f_v = model(inputs, False)
        momen_u = u_t + u*u_x + v*u_y - Pr*(u_xx + u_yy) + f_u
        momen_v = v_t + u*v_x + v*v_y - Pr*(v_xx + v_yy) + f_v
        con_u = u_x + u_y
        con_v = v_x + v_y
        momen_mse = (loss_fun(momen_u, momen_u.detach() * 0) + loss_fun(momen_v, momen_v.detach() * 0))/2
        con_mse = (loss_fun(con_u, con_u.detach() * 0) + loss_fun(con_v, con_v.detach() * 0))/2
        Loss = (momen_mse + con_mse)*0.5
        #
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
    
    
    return np.round(np.sqrt(np.mean(mses, axis = 0)),3)

# def train_epoch2(train_loader, model, optimizer, loss_fun):
#     mses = []
#     Pr = 0.71  

#     for inputs, u_target, v_target in train_loader:
#         inputs = inputs.to(device)
#         u_target = u_target.to(device)
#         v_target = v_target.to(device)
#         Loss = 0
#         u, v, f_u, f_v = model(inputs, False)
#         velocity_mse = (loss_fun(u, u_target) + loss_fun(v, v_target))/2
#         momen_mse = (loss_fun(f_u, f_u.detach() * 0) + loss_fun(f_v, f_v.detach() * 0))/2
#         Loss = velocity_mse+ momen_mse
#         mses.append(velocity_mse.item())
#         optimizer.zero_grad()
#         Loss.backward()
#         optimizer.step()
    
#     return np.round(np.sqrt(np.mean(mses, axis = 0)),3)


def eval_epoch(valid_loader, model, loss_fun):
    with torch.no_grad():
        preds_u, preds_v = [],[]
        mses = []
        for inputs, u_target, v_target in valid_loader:
            inputs = inputs.to(device)
            u_target = u_target.to(device)
            v_target = v_target.to(device)
            u, v = model(inputs, True)
            mse = (loss_fun(u, u_target).item() + loss_fun(v, v_target).item())/2
            mses.append(mse)
            
    return np.round(np.sqrt(np.mean(mses)),3)


def test_epoch(test_loader, model, loss_fun):
    with torch.no_grad():
        Us, Vs = [],[]
        valid_mse = []
        for inputs, u_target, v_target in test_loader:
            inputs = inputs.to(device)
            u_target = u_target.to(device)
            v_target = v_target.to(device)
            u, v = model(inputs, True)
            mse = (loss_fun(u, u_target).item() + loss_fun(v, v_target).item())/2
            valid_mse.append(mse)

            Us.append(torch.cat([inputs, u, u_target], dim = 1).cpu().data.numpy())
            Vs.append(torch.cat([inputs, v, v_target], dim = 1).cpu().data.numpy())

        Us = np.concatenate(Us, axis = 0)  
        Vs = np.concatenate(Vs, axis = 0)  
        num_edd = Us[:,-1].shape[0]//64//448

        u_pred = Us[:,-2].reshape(num_edd, -1)
        v_pred = Vs[:,-2].reshape(num_edd, -1)
        u_true = Us[:,-1].reshape(num_edd, -1)
        v_true = Vs[:,-1].reshape(num_edd, -1)

        loss_curve = np.sqrt((np.mean((u_pred[:60] - u_true[:60])**2, axis = 1) + np.mean((v_pred[:60] - v_true[:60])**2, axis = 1))/2)

    return {"preds": np.array([u_pred.reshape(num_edd, 64, 448), v_pred.reshape((num_edd, 64, 448))]), 
            "trues": np.array([u_true.reshape(num_edd, 64, 448), v_true.reshape((num_edd, 64, 448))]),
            "loss_curve": np.round(loss_curve,3)}