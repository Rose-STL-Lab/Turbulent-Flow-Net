###### Physics-Informed Neural Nets For Navier-Stokes Equation ######
# Maziar et al. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational physics 2019

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd 
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NN(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim, num_layers):
        super(NN, self).__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.input_layer =  nn.Linear(input_dim, hidden_dim)      
        self.middle_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
            
    def forward(self, x):
        out = torch.sin(self.input_layer(x))
        
        for i in range(self.num_layers - 2):
            out = torch.sin(self.middle_layer(out))
            
        out = self.output_layer(out)
        return out
    
class DHPM(nn.Module):
    def __init__(self, hidden_dim = [200,200], num_layers = [3,3]):
        super(DHPM, self).__init__()
        self.NN_u = NN(hidden_dim = hidden_dim[0], input_dim = 3, output_dim = 1, 
                       num_layers = num_layers[0]).to(device)
        
        self.NN_v = NN(hidden_dim = hidden_dim[0], input_dim = 3, output_dim = 1, 
                       num_layers = num_layers[0]).to(device)
        
        self.NN_f = NN(hidden_dim = hidden_dim[1], input_dim = 12, output_dim = 2, 
                       num_layers = num_layers[1]).to(device)
        
    def forward(self, inputs, test = False):
        inp_x = Variable(inputs[:,:1], requires_grad=True).to(device)
        inp_y = Variable(inputs[:,1:2], requires_grad=True).to(device)
        inp_t = Variable(inputs[:,-1:], requires_grad=True).to(device)
        
        u = self.NN_u(torch.cat([inp_x, inp_y, inp_t], dim = 1).to(device))
        v = self.NN_v(torch.cat([inp_x, inp_y, inp_t], dim = 1).to(device))
        
        
        if test:
            return u, v
        
        u_x = autograd.grad(outputs=u, inputs=inp_x, 
                            grad_outputs=torch.ones(u.size()).to(device),
                            create_graph=True)[0]
        u_y = autograd.grad(outputs=u, inputs=inp_y, 
                            grad_outputs=torch.ones(u.size()).to(device),
                            create_graph=True)[0] 
        u_t = autograd.grad(outputs=u, inputs=inp_t, 
                            grad_outputs=torch.ones(u.size()).to(device),
                            create_graph=True)[0]
        u_xx = autograd.grad(outputs=u_x, inputs=inp_x, 
                            grad_outputs=torch.ones(u_x.size()).to(device),
                            create_graph=True)[0]
        u_yy = autograd.grad(outputs=u_y, inputs=inp_y, 
                            grad_outputs=torch.ones(u_x.size()).to(device),
                            create_graph=True)[0]
        
        
        v_x = autograd.grad(outputs=v, inputs=inp_x, 
                            grad_outputs=torch.ones(v.size()).to(device),
                            create_graph=True)[0]
        v_y = autograd.grad(outputs=v, inputs=inp_y, 
                            grad_outputs=torch.ones(v.size()).to(device),
                            create_graph=True)[0]
        v_t = autograd.grad(outputs=v, inputs=inp_t, 
                            grad_outputs=torch.ones(v.size()).to(device),
                            create_graph=True)[0]
        v_xx = autograd.grad(outputs=v_x, inputs=inp_x, 
                            grad_outputs=torch.ones(v_x.size()).to(device),
                            create_graph=True)[0]
        v_yy = autograd.grad(outputs=v_y, inputs=inp_y, 
                            grad_outputs=torch.ones(v_y.size()).to(device),
                            create_graph=True)[0]
        
        f = self.NN_f(torch.cat([u, v, u_t, v_t, u_x, u_xx, u_y, u_yy, v_x, v_xx, v_y, v_yy], dim = 1).to(device))
        f_u = f[:,:1]
        f_v = f[:,1:]
        
        return u, v, u_t, v_t, u_x, u_xx, u_y, u_yy, v_x, v_xx, v_y, v_yy, f_u, f_v
