import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torch.utils import data
import time
import random
from models import TFNet, DivergenceLoss
from utils import Dataset, train_epoch, eval_epoch, test_epoch, divergence, spectrum_band
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_direc = "rbc_data/sample_"
test_direc = "rbc_data/sample_"

parser = argparse.ArgumentParser(description='Approximately Equivariant CNNs')
parser.add_argument('--kernel_size', type=int, required=False, default="3", help='convolution kernel size')
parser.add_argument('--time_range', type=int, required=False, default="6", help='moving average window size for temporal filter')
parser.add_argument('--output_length', type=int, required=False, default="4", help='number of prediction losses used for backpropagation')
parser.add_argument('--input_length', type=int, required=False, default="25", help='input length')
parser.add_argument('--batch_size', type=int, required=False, default="32", help='batch size')
parser.add_argument('--num_epoch', type=int, required=False, default="1000", help='maximum number of epochs')
parser.add_argument('--learning_rate', type=float, required=False, default="0.001", help='learning rate')
parser.add_argument('--decay_rate', type=float, required=False, default="0.95", help='learning decay rate')
parser.add_argument('--dropout_rate', type=float, required=False, default="0.0", help='dropout rate')
parser.add_argument('--coef', type=float, required=False, default="0.0", help='the coefficient for divergence free regularizer')
parser.add_argument('--inp_dim', type=int, required=False, default="2", help='number of channels per frames')
parser.add_argument('--seed', type=int, required=False, default="0", help='random seed')
args = parser.parse_args()

random.seed(args.seed)  # python random generator
np.random.seed(args.seed)  # numpy random generator

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


time_range = args.time_range
output_length = args.output_length
input_length = args.input_length
learning_rate = args.learning_rate
dropout_rate = args.dropout_rate
kernel_size = args.kernel_size
batch_size = args.batch_size
num_epoch = args.num_epoch
coef = args.coef
decay_rate = args.decay_rate
inp_dim = args.inp_dim

model_name = "TFNet_seed{}_bz{}_inp{}_pred{}_lr{}_decay{}_coef{}_dropout{}_kernel{}_win{}".format(args.seed,
                                                                                                  batch_size,
                                                                                                  input_length,
                                                                                                  output_length,
                                                                                                  learning_rate,
                                                                                                  decay_rate,
                                                                                                  coef,
                                                                                                  dropout_rate,
                                                                                                  kernel_size, 
                                                                                                  time_range)
                                                                                     
                                                                                              
# train-valid-test split
train_indices = list(range(0, 6000))
valid_indices = list(range(6000, 7700))
test_indices = list(range(7700, 9800))

model = TFNet(input_channels = input_length*inp_dim, 
              output_channels = inp_dim, 
              kernel_size = kernel_size, 
              dropout_rate = dropout_rate, 
              time_range = time_range).to(device)

train_set = Dataset(train_indices, input_length + time_range - 1, 40, output_length, train_direc, True)
valid_set = Dataset(valid_indices, input_length + time_range - 1, 40, 6, test_direc, True)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 2)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 2)


loss_fun = torch.nn.MSELoss()
regularizer = DivergenceLoss(torch.nn.MSELoss())
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = decay_rate)

train_rmse = []
valid_rmse = []
test_rmse = []
min_rmse = 1e8
for i in range(num_epoch):
    start = time.time()
    torch.cuda.empty_cache()
    
    model.train()
    train_rmse.append(train_epoch(train_loader, model, optimizer, loss_fun, coef, regularizer))
    
    model.eval()
    rmse, preds, trues = eval_epoch(valid_loader, model, loss_fun)
    valid_rmse.append(rmse)
    
    if valid_rmse[-1] < min_rmse:
        min_rmse = valid_rmse[-1] 
        best_model = model
    end = time.time()
    
    # Early stopping 
    if (len(train_rmse) > 100 and np.mean(valid_rmse[-5:]) >= np.mean(valid_rmse[-10:-5])):
            break
            
    print("Epoch {} | T: {:0.2f} | Train RMSE: {:0.3f} | Valid RMSE: {:0.3f}".format(i+1, (end-start)/60, train_rmse[-1], valid_rmse[-1]))
    scheduler.step()


# Testing
test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, test_direc, True)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 2)
test_preds, test_trues, rmse_curve = test_epoch(test_loader, best_model, loss_fun)

# Denormalization: Optional
mean_vec = np.array([-1.6010, 0.0046]).reshape(1, 1, 2, 1, 1) 
norm_std = 2321.9727
test_preds = test_preds * norm_std + mean_vec
test_trues = test_trues * norm_std + mean_vec

# Compute evaluation scores
rmse_curve = np.sqrt(np.mean((test_preds - test_trues)**2, axis = (0,2,3,4)))
div_curve = divergence(test_preds)
energy_spectrum = spectrum_band(test_preds)

torch.save({"test_preds": test_preds[::60],
            "test_trues": test_trues[::60],
            "rmse_curve": rmse_curve, 
            "div_curve": div_curve, 
            "spectrum": energy_spectrum}, 
            model_name + "pt")
