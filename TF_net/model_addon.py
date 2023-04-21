import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
# from positional_encodings.torch_encodings import PositionalEncoding1D
from torch.utils import data
import math

# def get_time_emb_backbone(tstep, xx_len, test_mode, w=64, h=64):
#     inp_len = xx_len // 2

#     def single_channel(tstep, inp_len, dim, b_size=1):
#         # b_size is irrelavant, the value will be same for each batch dimension
#         p_enc_1d_model = PositionalEncoding1D(dim)
#         emb_full =  p_enc_1d_model(torch.rand(b_size, tstep + inp_len, dim))
#         return emb_full[:,-inp_len:]

#     single_ch_emb = single_channel(tstep, inp_len, h*w, 1)
#     # we want concatenation to be alternate: https://stackoverflow.com/questions/61026393/pytorch-concatenate-rows-in-alternate-order
#     # After operation, emb[0,i] == emb[0,i+1] where i is even
#     emb = torch.cat((single_ch_emb, single_ch_emb), axis=-1).reshape((1, -1, h, w))  
#     if test_mode > 1:
#         emb = [emb for _ in range(test_mode)]
#         emb = torch.cat(emb, dim=-1)
#     return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        freqs = x * self.weights.unsqueeze(dim=0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
    
class TimeEmbMLP(nn.Module):
    def __init__(self, time_emb_dim, dim_out):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        )
        for k in self.mlp:
            if isinstance(k, nn.Linear):
                k.weight.data.fill_(0.0)
                k.bias.data.fill_(0.0)
    
    def forward(self, x, time_emb):
        time_emb = self.mlp(time_emb)
        time_emb = time_emb[..., None, None]
        scale_shift = time_emb.chunk(2, dim = 1)
        scale, shift = scale_shift
        x = x * (scale + 1) + shift
        return x

def conv(input_channels, output_channels, kernel_size, stride, dropout_rate, pad=None):
    pad = (kernel_size - 1) // 2 if pad is None else pad
    layer = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, 
                  stride = stride, padding=pad),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_rate)
    )
    return layer

def deconv(input_channels, output_channels):
    layer = nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True)
    )
    return layer

def deconv_conv2d(input_channels, output_channels):
    layer = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding='same'),
        nn.LeakyReLU(0.1, inplace=True)
    )
    return layer


    
class Encoder(nn.Module):
    def __init__(self, input_channels, kernel_size, dropout_rate, addon, time_emb_dim):
        super(Encoder, self).__init__()
        self.addon = addon
        self.conv1 = conv(input_channels, 64, kernel_size=kernel_size, stride = 2, dropout_rate = dropout_rate)
        if self.addon >= 1: self.conv12 = conv(64, 64, kernel_size=kernel_size, stride = 1, dropout_rate = dropout_rate, pad='same')
        self.conv2 = conv(64, 128, kernel_size=kernel_size, stride = 2, dropout_rate = dropout_rate)
        if self.addon >= 2: self.conv23 = conv(128, 128, kernel_size=kernel_size, stride = 1, dropout_rate = dropout_rate, pad='same')
        self.conv3 = conv(128, 256, kernel_size=kernel_size, stride = 2, dropout_rate = dropout_rate)
        if self.addon >= 3: self.conv34 = conv(256, 256, kernel_size=kernel_size, stride = 1, dropout_rate = dropout_rate, pad='same')
        self.conv4 = conv(256, 512, kernel_size=kernel_size, stride = 2, dropout_rate = dropout_rate)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.002/n)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.time_emb_dim=time_emb_dim
        if self.time_emb_dim:
            self.time_emb_mlp1 = TimeEmbMLP(time_emb_dim, 64)
            self.time_emb_mlp2 = TimeEmbMLP(time_emb_dim, 128)
            self.time_emb_mlp3 = TimeEmbMLP(time_emb_dim, 256)
            self.time_emb_mlp4 = TimeEmbMLP(time_emb_dim, 512)
                
    def forward(self, x, time_emb=None):
        out_conv1 = self.conv1(x)
        if self.addon>=1: out_conv1 = self.conv12(out_conv1)
        if self.time_emb_dim: out_conv1 = self.time_emb_mlp1(out_conv1, time_emb)

        out_conv2 = self.conv2(out_conv1)
        if self.addon>=2: out_conv2 = self.conv23(out_conv2)
        if self.time_emb_dim: out_conv2 = self.time_emb_mlp2(out_conv2, time_emb)
        
        out_conv3 = self.conv3(out_conv2)
        if self.addon>=3: out_conv3 = self.conv34(out_conv3)
        if self.time_emb_dim: out_conv3 = self.time_emb_mlp3(out_conv3, time_emb)
        
        out_conv4 = self.conv4(out_conv3)
        if self.time_emb_dim: out_conv4 = self.time_emb_mlp4(out_conv4, time_emb)
        
        return out_conv1, out_conv2, out_conv3, out_conv4 


class LES(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate, time_range, addon_enc=False, addon_dec=False, time_emb_dim=0):
        super(LES, self).__init__()
        self.spatial_filter = nn.Conv2d(1, 1, kernel_size = 3, padding = 1, bias = False)   
        self.temporal_filter = nn.Conv2d(time_range, 1, kernel_size = 1, padding = 0, bias = False)
        self.input_channels = input_channels
        self.time_range = time_range

        self.time_emb_dim = time_emb_dim
        if time_emb_dim:
            self.time_emb_backbone = nn.Sequential(
                LearnedSinusoidalPosEmb(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim*4),
                nn.GELU(),
                nn.Linear(time_emb_dim*4, time_emb_dim*4)
            )
            
            time_emb_dim*=4
            self.time_emb_dim = time_emb_dim

            self.time_emb_mlp0 = TimeEmbMLP(time_emb_dim, 32)
            self.time_emb_mlp1 = TimeEmbMLP(time_emb_dim, 64)
            self.time_emb_mlp2 = TimeEmbMLP(time_emb_dim, 128)
            self.time_emb_mlp3 = TimeEmbMLP(time_emb_dim, 256)
        else:
            self.time_emb_backbone = None
        
        self.encoder1 = Encoder(input_channels, kernel_size, dropout_rate, addon_enc, time_emb_dim)
        self.encoder2 = Encoder(input_channels, kernel_size, dropout_rate, addon_enc, time_emb_dim)
        self.encoder3 = Encoder(input_channels, kernel_size, dropout_rate, addon_enc, time_emb_dim)

        self.addon_dec = addon_dec
        
        self.deconv3 = deconv(512, 256)
        if self.addon_dec>=1: self.deconv32 = deconv_conv2d(256, 256)
        self.deconv2 = deconv(256, 128)
        if self.addon_dec>=2: self.deconv21 = deconv_conv2d(128, 128)
        self.deconv1 = deconv(128, 64)
        if self.addon_dec>=3: self.deconv10 = deconv_conv2d(64, 64)
        self.deconv0 = deconv(64, 32)
        self.output_layer = nn.Conv2d(32 + input_channels, output_channels, kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2)
        
    def forward(self, xx, test_mode=False, tstep=None):
        #print("shape:",xx.shape)
        xx_len = xx.shape[1]

        width = 64 if not test_mode else 64*7
        # u = u_mean + u_tilde + u_prime
        u_tilde = self.spatial_filter(xx.reshape(xx.shape[0]*xx.shape[1], 1, 64, width)).reshape(xx.shape[0], xx.shape[1], 64, width)
        # u_prime
        u_prime = (xx - u_tilde)[:,(xx_len - self.input_channels):]
        # u_mean
        u_tilde2 = u_tilde.reshape(u_tilde.shape[0], u_tilde.shape[1]//2, 2, 64, width)
        u_mean = []
        for i in range(xx_len//2 - self.input_channels//2, xx_len//2):
            cur_mean = torch.cat([self.temporal_filter(u_tilde2[:,i-self.time_range+1:i+1,0,:,:]).unsqueeze(2), 
                                  self.temporal_filter(u_tilde2[:,i-self.time_range+1:i+1,1,:,:]).unsqueeze(2)], dim = 2)
            u_mean.append(cur_mean)
        u_mean = torch.cat(u_mean, dim = 1)
        u_mean = u_mean.reshape(u_mean.shape[0], -1, 64, width)
        # u_tilde
        u_tilde = u_tilde[:,(self.time_range-1)*2:] - u_mean
        #print("u_mean:",u_mean.shape,"u_tilde:",u_tilde.shape,"u_prime:",u_prime.shape)
        #xxxx = 1
        #assert xxxx > 10,"stop"

        assert (self.time_emb_dim > 0) == (tstep is not None)
        if self.time_emb_dim:
            time_emb = self.time_emb_backbone(torch.tensor([tstep for _ in range(xx.shape[0])]).to(xx.device))
        else:
            time_emb=None

        out_conv1_mean, out_conv2_mean, out_conv3_mean, out_conv4_mean = self.encoder1(u_mean, time_emb)
        out_conv1_tilde, out_conv2_tilde, out_conv3_tilde, out_conv4_tilde = self.encoder2(u_tilde, time_emb)
        out_conv1_prime, out_conv2_prime, out_conv3_prime, out_conv4_prime = self.encoder3(u_prime, time_emb)
        
        out_deconv3 = self.deconv3(out_conv4_mean + out_conv4_tilde + out_conv4_prime)
        if self.addon_dec>=1: out_deconv3 = self.deconv32(out_deconv3)
        if self.time_emb_dim: out_deconv3 = self.time_emb_mlp3(out_deconv3, time_emb)

        out_deconv2 = self.deconv2(out_conv3_mean + out_conv3_tilde + out_conv3_prime + out_deconv3)
        if self.addon_dec>=2: out_deconv2 = self.deconv21(out_deconv2)
        if self.time_emb_dim: out_deconv2 = self.time_emb_mlp2(out_deconv2, time_emb)

        out_deconv1 = self.deconv1(out_conv2_mean + out_conv2_tilde + out_conv2_prime + out_deconv2)
        if self.addon_dec>=3: out_deconv1 = self.deconv10(out_deconv1)
        if self.time_emb_dim: out_deconv1 = self.time_emb_mlp1(out_deconv1, time_emb)

        out_deconv0 = self.deconv0(out_conv1_mean + out_conv1_tilde + out_conv1_prime + out_deconv1)
        if self.time_emb_dim: out_deconv0 = self.time_emb_mlp0(out_deconv0, time_emb)

        concat0 = torch.cat((xx[:,(xx_len - self.input_channels):], out_deconv0), 1)
        out = self.output_layer(concat0)
        #print("output:",out.shape)
        #xxxx = 1
        #assert xxxx > 10,"stop"
        return out
  
