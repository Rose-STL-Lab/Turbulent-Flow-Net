###### A generative adversarial neural nets with a U-net generator and Convolution discriminator.######
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size - 1) // 2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, inplace=True),
    )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                           stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True),
    )

def predict_flow(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

class U_net(nn.Module):
    def __init__(self, input_channels=4, output_channels=1):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = conv(input_channels, 64, kernel_size=3, stride=2)
        self.conv2 = conv(64, 128, kernel_size=3, stride=2)
        self.conv3 = conv(128, 256, kernel_size=3, stride=2)
        self.conv3_1 = conv(256, 256, kernel_size=3)
        self.conv4 = conv(256, 512, kernel_size=3, stride=2)
        self.conv4_1 = conv(512, 512, kernel_size=3)
        self.conv5 = conv(512, 1024, kernel_size=3, stride=2)
        #self.conv5_1 = conv(1024, 1024)

        self.deconv4 = deconv(1024, 256)
        self.deconv3 = deconv(768, 128)
        self.deconv2 = deconv(384, 64)
        self.deconv1 = deconv(192, 32)
        self.deconv0 = deconv(96, 16)
    
        self.predict_flow0 = predict_flow(16 + input_channels, output_channels)

    def forward(self, x):

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5(out_conv4)

        out_deconv4 = self.deconv4(out_conv5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        flow0 = self.predict_flow0(concat0)

        return flow0
    
class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()
        self.model = U_net(input_channels = input_channels, output_channels = output_channels)
        
    def forward(self, xx, output_steps):
        ims = []
        for i in range(output_steps):
            im = self.model(xx)
            ims.append(im)
            xx = torch.cat([xx[:, 2:], im], 1)
        return torch.cat(ims, dim = 1)
    
class Discriminator_Spatial(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator_Spatial, self).__init__()
        self.activ = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size = 5, padding = 2, stride = 2),
            nn.BatchNorm2d(32)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 5, padding = 2, stride = 2),
            nn.BatchNorm2d(64)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 5, padding = 2, stride = 2),
            nn.BatchNorm2d(128)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(256)
        )
        
        self.dense_layer = nn.Sequential(
            nn.Linear(256*8*8, 1),
            nn.Sigmoid()
        )

    def forward(self, ims):
        ims1 = self.activ(self.conv1(ims))
        ims2 = self.activ(self.conv2(ims1))
        ims3 = self.activ(self.conv3(ims2))
        ims4 = self.activ(self.conv4(ims3))
        
        out = ims4.reshape(ims4.shape[0], -1)
        out = self.dense_layer(out)
        return out,  Variable(ims1, requires_grad=True), Variable(ims2, requires_grad=True), Variable(ims3, requires_grad=True), Variable(ims4, requires_grad=True)
