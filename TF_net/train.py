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
from collections import deque
import cv2 as cv
from sklearn.mixture import GaussianMixture

class EMA:
    def __init__(self, _max = 5) -> None:
        self.Q = deque()
        self._max = _max

    def append(self, x):
        if len(self.Q) >= self._max:
            self.Q.popleft()
        self.Q.append(x)
    
    def mean(self):
        return sum(self.Q) / len(self.Q)

class customExpLR:
    def __init__(self, optimizer, init_lr, tmax, gamma, restart_factor= 0.1) -> None:
        self.init_lr = init_lr
        self.optimizer = optimizer
        self.tmax = tmax
        self.gamma = gamma
        self.restart_factor = restart_factor
        
    
    def step(self, epoch) -> None:
        if (epoch+1) % self.tmax == 0:
            for grp in self.optimizer.param_groups:
                self.init_lr = self.init_lr*self.restart_factor
                grp['lr'] = self.init_lr*self.restart_factor
        else:
            for grp in self.optimizer.param_groups:
                grp['lr'] *= self.gamma
        return

class Scaler:
    def __init__(self, type, offset=0, over_ch = True):
        type_ch = ['std','norm', 'log', 'none']
        if type not in type_ch:
            raise ValueError(f"{type} is not in {type_ch}")
        self.type = type
        self.over_ch = over_ch
        self.offset = offset

    def fit_transform(self, x):
        assert self.over_ch or x.shape[1] == 2
        dim = tuple(range(len(x.shape))) if self.over_ch else (0,) + tuple(range(2,len(x.shape)))
        keepdim = False if self.over_ch else True
        identity = lambda x: x

        # apply offset
        x = x[self.offset:]

        # Consider types
        if self.type == 'std':
            self.alpha = torch.mean(x, dim=dim, keepdim=keepdim)
            self.beta = torch.std(x, dim=dim, keepdim=keepdim)
            self.fun = self.inv_fun = identity
        elif self.type == 'norm':
            _max= torch.amax(x, dim=dim, keepdim=keepdim)
            _min= torch.amin(x, dim=dim, keepdim=keepdim)
            self.alpha = _min
            self.beta = _max - _min
            self.fun = self.inv_fun = identity
        elif self.type == 'log':
            _min= torch.amin(x, dim=dim, keepdim=keepdim)
            self.alpha = _min - 1
            self.beta = _min/_min #cheap trick for getting 1
            self.fun = torch.log
            self.inv_fun = torch.exp
        elif self.type == 'none':
            self.alpha = torch.tensor(0)
            self.beta = torch.tensor(1)
            self.fun = self.inv_fun = identity
        else:
            raise ValueError("Dude, update me with new types!!!")

        assert self.beta == 1 or self.fun == identity   #This assertion since at test time, printing loss curve without inv transform multiplies with beta
        print(self.alpha.view(-1)); print(self.beta.view(-1), self.fun, self.inv_fun)
        return self.fun(x - self.alpha)/self.beta

    def inv_transform(self, y):
        assert self.over_ch or y.shape[1] == 2
        if type(y) is not np.ndarray:
            self.beta = self.beta.to(y.device)
            self.alpha = self.alpha.to(y.device)
        else:
            y = torch.from_numpy(y)
        
        out = self.inv_fun(y * self.beta) + self.alpha
        return out if type(y) is not np.ndarray else out.numpy()

def preprocess(args, permute = False, compress = True, test_mode=False, get_opt_flow=False):
    data = torch.load(args.data)
    if get_opt_flow: opt_flow = torch.load(args.data.replace('.pt','_opt_flow.pt'))

    if permute:
        data = torch.permute(data, (0, 3, 1, 2))

    if compress:
        data = data[:,:,::4,::4]
        if get_opt_flow: opt_flow = opt_flow[:,:,::4,::4]

    data = args.transform.fit_transform(data)

    # divide each rectangular snapshot into 7 subregions
    # data shape: num_subregions * time * channels * w * h
    if not test_mode:
        data = torch.stack([data[:,:,:,k*64:(k+1)*64] for k in range(7)])
        if get_opt_flow: opt_flow = torch.stack([opt_flow[:,:,:,k*64:(k+1)*64] for k in range(7)])
        #print(data.shape)
    data = torch.FloatTensor(data) # full domain
    if get_opt_flow: opt_flow = opt_flow.float()
    if get_opt_flow: return data, opt_flow 
    else: return data

class Dataset(data.Dataset):
    def __init__(self, indices, input_length, mid, output_length, data_prep, stack_x, test_mode=False, test_mode_train=False, split_spatially=False,
                    noise=0.0, do_not_scale_noise=False, opt_flow=None): # test_mode: full areas or not
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.stack_x = stack_x
        self.data_prep = data_prep
        self.list_IDs = indices
        self.test_mode = test_mode
        self.noise = noise
        self.test_mode_train = test_mode_train
        self.do_not_scale_noise = do_not_scale_noise
        self.opt_flow = opt_flow

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        j = ID // 7 if not self.test_mode_train else ID
        #i = ID % 7
        if self.test_mode or self.test_mode_train:
            data_ = self.data_prep[j:j+100]#[i,j:j+100]
            if self.opt_flow is not None: 
                opt_flow_ = self.opt_flow[j:j+100]
        else:
            i = ID % 7
            data_ = self.data_prep[i,j:j+100]
            if self.opt_flow is not None: 
                opt_flow_ = self.opt_flow[i,j:j+100]

        y = data_[self.mid:(self.mid+self.output_length)]
        if self.opt_flow is not None: opt_flow_ = opt_flow_[self.mid:(self.mid+self.output_length)]

        if self.stack_x:
            x = data_[(self.mid-self.input_length):self.mid].reshape(-1, y.shape[-2], y.shape[-1])
        else:
            x = data_[(self.mid-self.input_length):self.mid]
        
        if self.do_not_scale_noise:
            x = x + self.noise*0.01*torch.randn_like(x)
        else:
            x = x*(1 + self.noise*0.01*torch.randn_like(x))
        
        if self.opt_flow is not None: return x.float(), y.float(), opt_flow_
        else: return x.float(), y.float()
    
def mask_gen(epoch, start, end, lower, upper, warmup, tile_sz=4, image_sz=64):
    assert image_sz%tile_sz == 0    #code not tested for other cases
    mask_ratio = 1 if epoch < warmup else 0.01*min(upper, max(lower, lower + ((upper-lower)*(epoch-start))/(end-start)))
    iv,jv = [x.flatten() for x in np.meshgrid(np.arange(image_sz//tile_sz), np.arange(image_sz//tile_sz), indexing='ij')]
    mask_i = np.random.choice(len(iv), int(mask_ratio*len(iv)), replace=False)

    mask = torch.ones((image_sz,image_sz))
    for idx in zip(mask_i):
        i, j = iv[idx]*tile_sz, jv[idx]*tile_sz
        mask[i:i+tile_sz,j:j+tile_sz] = 0
    # print(mask_ratio, (mask == 0).sum() / len(mask.flatten()))
    return mask

def mask_gen_opt(epoch, opt_flow_mag, start, end, lower, upper, warmup, tile_sz=1, image_sz=64):
    b,w,h = opt_flow_mag.shape
    assert w==h==image_sz
    assert tile_sz == 1 #at the moment only tile_sz 1 supported
    assert image_sz%tile_sz == 0    #code not tested for other cases
    mask_ratio = 1 if epoch < warmup else 0.01*min(upper, max(lower, lower + ((upper-lower)*(epoch-start))/(end-start)))
    thresh = torch.quantile(opt_flow_mag.reshape((b,-1)), mask_ratio, dim=1).reshape((b,1,1))
    mask = torch.ones_like(opt_flow_mag)
    mask[opt_flow_mag <= thresh] = 0
    # print(epoch, mask_ratio, (mask == 0).sum() / len(mask.flatten()))
    return mask

class GMM:
    def __init__(self, loss, n_comps, means_init=None, ignore_min2=False):
        loss = loss.reshape((-1,1))
        self.gmm = GaussianMixture(n_components=n_comps, random_state=0, init_params='k-means++', means_init=means_init).fit(loss.cpu().detach())
        self.means_ = self.gmm.means_
        self.covariances_ = self.gmm.covariances_
        self.max_ndx = np.argmax(self.gmm.means_, axis=0)[0]
        self.min_ndx = np.argmin(self.gmm.means_, axis=0)[0]
        self.ignore_min2 = ignore_min2

    def convert(self, loss):
        shp = loss.shape
        loss = loss.reshape((-1,1))
        detached_loss = loss.cpu().detach()
        probs=1 - self.gmm.predict_proba(detached_loss)[:,self.max_ndx] - (self.gmm.predict_proba(detached_loss)[:,self.min_ndx] if self.ignore_min2 else 0)
        # print("fraction of probs > 0.001: ", 100*((np.sum(probs > 0.1)) / len(probs)))
        loss = loss*torch.tensor(probs[:,None]).to(loss.device)
        return loss.reshape(shp)
    
def train_epoch(args, train_loader, model, optimizer, loss_function, m_pred= None, coef = 0, regularizer = None, coef2=1.0, epoch=-1,barrier=1e2,mide=None,slope=None, 
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if slope is None:
        slope = m_pred.slope
    train_mse = []
    train_reg = []
    training_data = tqdm(train_loader)
    gmm = None
    gmm_train_batch = lambda x: (x==0)
    for batch_idx, batch_data in enumerate(training_data):
        if args.mask and args.mtype=='opt':
            xx, yy, opt_flow = batch_data
            opt_flow = opt_flow.transpose(0,1)
        else:
            xx, yy = batch_data
        loss = 0
        batch_reg = 0
        ims = []
        xx = xx.to(device) # [batch,time,width,height]
        yy = yy.to(device).transpose(0,1)
        length = len(yy) if args.trunc > len(yy) else args.trunc
        norm_trunc = len(yy) - args.trunc + 1
        prev_lya = None # for approximating dV/dt~V(t+1)-V(t)
        pred_losses = []

        # Auto-regressive gen
        for cur_t, y in enumerate(yy): 
            if args.mask:
                if args.mtype=='random':
                    predict_mask = mask_gen(epoch, args.mstart, args.mend, args.mlower, args.mupper, args.mtile).to(xx.device)
                elif args.mtype=='opt':
                    predict_mask = mask_gen_opt(epoch, opt_flow[cur_t], args.mstart, args.mend, args.mlower, args.mupper, args.mwarmup, args.mtile).to(xx.device)
                    predict_mask = predict_mask.unsqueeze(dim=1)
                loss_mask = 1 if args.mfloss else (1-predict_mask)
                if args.mcurr:
                    predict_mask = 0
            else:
                loss_mask = 1
            inp = torch.cat((xx[:,2:], y*predict_mask), 1) if args.mask else xx
            im = model(inp, tstep = cur_t if args.pos_emb else None)
            if args.teacher_forcing:
                xx = torch.cat([xx[:, 2:], y], 1)
            else:
                xx = torch.cat([xx[:, 2:], im], 1)
            pred_loss = (loss_function(im, y)*loss_mask)
            if cur_t == args.trunc-1:
                trunc_thresh = torch.max(pred_loss)
                trunc_pred_loss_mean = pred_loss.mean().clone().detach()
                pred_loss /= norm_trunc
            elif cur_t >= args.trunc:
                about_to_cut = torch.sum(pred_loss > trunc_thresh*args.trunc_factor)
                pred_loss[pred_loss > trunc_thresh*args.trunc_factor] = 0
                pred_loss = pred_loss * (trunc_pred_loss_mean/pred_loss.mean())
                pred_loss /= norm_trunc
            if args.gmm_comp > 0 and gmm is not None:
                pred_loss = gmm.convert(pred_loss)
            if gmm_train_batch(batch_idx):
                pred_losses.append(pred_loss[0])    # Only first element of first batch is used for training
            pred_loss = pred_loss.mean()
            pred_loss = pred_loss * args.beta**cur_t
            lya_val = lyapunov_func(im,y)  # (Batch, )
            if coef != 0:
                loss += pred_loss + coef*regularizer(im, y)
            else:
                loss += pred_loss
                
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
        if (args.gmm_comp > 0) and gmm_train_batch(batch_idx):
            gmm = GMM(torch.stack(pred_losses), args.gmm_comp, means_init = None if gmm is None else gmm.means_, \
                      ignore_min2=args.ignore_min2 if epoch > args.ignore_min2_epoch else False)
        ims = np.concatenate(ims, axis = 1)
        train_mse.append(loss.item()/length) 
        train_reg.append(batch_reg/length)
        optimizer.zero_grad()
        if args.rescale_loss: 
            loss = loss*4/length
        if args.norm_loss:
            loss = loss/length
        loss.backward()
        optimizer.step()
        if coef2 != 0:
            if log_c + relu_c > 0:
                training_data.set_postfix(epoch=epoch,log_p=log_c/(log_c+relu_c),log_v=log_v,relu_v=relu_v)
            else:
                raise ValueError("Loss became NaN, probably!")
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    train_reg = round(np.mean(train_reg),5)
    return train_mse,train_reg

def eval_epoch(args, valid_loader, model, loss_function,coef2=1.0,barrier=1e2,mide=None,slope=None, device=None):
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
            
            for cur_t, y in enumerate(yy):
                inp = torch.cat((xx[:,2:], torch.zeros_like(y)), 1) if args.mask else xx
                im = model(inp, tstep = cur_t if args.pos_emb else None)
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

def test_epoch(args, test_loader, model, loss_function,test_mode=True, save_preds=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_batches=float('inf')):
    if args.inv_transform:
        print("==============Samples will be inverse transformed for correct estimates!================")
    valid_mse = []
    if save_preds:
        preds = []
        trues = []
    else:
        preds = trues = None
    with torch.no_grad():
        loss_curve = []
        test_data = tqdm(test_loader)
        for batch_no, (xx, yy) in enumerate(test_data):
            if batch_no >= num_batches:
                break
            xx = xx.to(device)
            yy = yy.to(device)
            
            loss = 0
            ims = []

            for cur_t, y in enumerate(yy.transpose(0,1)):
                try:
                    inp = torch.cat((xx[:,2:], torch.zeros_like(y)), 1) if args.mask else xx
                    # im = model(inp, test_mode=test_mode, tstep = min(cur_t, args.output_length-1) if args.pos_emb else None)
                    im = model(inp, test_mode=test_mode)
                except TypeError as err:
                    tqdm.write(f"{xx.shape}")
                    raise TypeError(err)
                xx = torch.cat([xx[:, 2:], im], 1)
                if args.inv_transform:
                    mse = loss_function(args.transform.inv_transform(im), args.transform.inv_transform(y))
                else:
                    mse = loss_function(im, y)
                loss += mse
                loss_curve.append(mse.item())
                
                ims.append(im.cpu().data.numpy())

            ims = np.array(ims).transpose(1,0,2,3,4)    
            if save_preds:
                preds.append(ims)
                trues.append(yy.cpu().data.numpy())            
            valid_mse.append(loss.item()/yy.shape[1])

        if save_preds:
            preds = np.concatenate(preds, axis = 0)  
            trues = np.concatenate(trues, axis = 0)
        
        valid_mse = round(np.mean(valid_mse), 5)
        loss_curve = np.array(loss_curve).reshape(-1,60)
        loss_curve = np.sqrt(np.mean(loss_curve, axis = 0))
        print(args.transform.beta.numpy() * loss_curve if not args.inv_transform else loss_curve)
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
        assert dV_dt.shape[0] == cur_pred_mse.shape[0] # "batch not match"
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
