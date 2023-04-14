from distutils.command.config import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import argparse
import copy
import numpy as np
from dse_duomotai_20230213 import SELayer2d_20230213 as att
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self,config):
    # def __init__(self,):
        super(UNet, self).__init__()
        self.n_channels = 3
        self.n_classes = 2
        self.bilinear =True
        std=4
        self.inc = DoubleConv(self.n_channels, std)
        self.down1 = Down(std, 2*std)
        self.down2 = Down(2*std, 4*std)
        self.down3 = Down(4*std,8*std)
        self.down4 = Down(8*std, 8*std)
        self.up1 = Up(16*std, 4*std, self.bilinear)
        self.up2 = Up(8*std, 2*std, self.bilinear)
        self.up3 = Up(4*std, std, self.bilinear)
        self.up4 = Up(2*std, std, self.bilinear)

        self.se1_down4=att(32,32)
        self.se2_down3=att(16,16)
        self.se3_down2=att(8,8)
        self.se4_down1=att(4,4)

        self.fc1 = nn.Sequential(
            nn.Linear(int(std*120*144), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)  # 默认就是0.5
        )
        self.fc2= nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.fc3= nn.Sequential(
            nn.Linear(4096, self.n_classes)
        )
        self.fc_list = [self.fc1, self.fc2, self.fc3]
        # self.head = nn.Linear(config.hidden_size,2)#768-->2

        print("Unet_T Model Initialize Successfully!")

    def forward(self, CSF,GM,WM):
    # def forward(self,x):#
        #1.特征融合
        x=torch.cat([CSF,GM,WM],dim=1)

        #2.初步特征提取
        #3.基础模块
        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

       

        x5 = self.down4(x4)


  
      
        #注意力模块
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)


       
        output = x.view(x.size()[0], -1) #[2, 4*224*224]
      
        for fc in self.fc_list:        # 3 FC
            output = fc(output)
     
        output=F.softmax(output)

   
        return output




