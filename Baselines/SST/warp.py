import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class DenseGridGen(nn.Module):
    def __init__(self, transpose=True):
        super(DenseGridGen, self).__init__()
        self.transpose = transpose
        self.register_buffer('grid', torch.Tensor())

    def forward(self, x):

        if self.transpose:
            x = x.transpose(1, 2).transpose(2, 3)
        
        g0 = torch.linspace(-1, 1, x.size(2)
                            ).unsqueeze(0).repeat(x.size(1), 1)
        
        g1 = torch.linspace(-1, 1, x.size(1)
                            ).unsqueeze(1).repeat(1, x.size(2))
        grid = torch.cat([g0.unsqueeze(-1), g1.unsqueeze(-1)], -1)
        self.grid.resize_(grid.size()).copy_(grid)

        bgrid = Variable(self.grid)
        bgrid = bgrid.unsqueeze(0).expand(x.size(0), *bgrid.size())

        return bgrid - x
    
class GaussianWarpingScheme(nn.Module):
    def __init__(self, padding_mode='zeros', std=0.25):
        super(GaussianWarpingScheme, self).__init__()
        self.grid = DenseGridGen()
        self.std = std
        self.padding_mode = padding_mode

    def forward(self, im, w):
        return F.grid_sample(im, self.grid(w), padding_mode=self.padding_mode, mode='bilinear')