import torch
from torch import nn
import numpy as np

class Unet2D(nn.Module):
    def __init__(self, in_channels, out_channels, channel_ratio=1):
        super().__init__()
        ch = np.array([32, 64, 128, 256, 512])
        ch = channel_ratio*ch
        ch = ch.round().astype(int)
        self.ch = ch




        self.conv1 = self.single_conv(in_channels, ch[0],5,2)
        self.conv2 = self.single_conv(ch[0], ch[1], 5,2)
        self.conv3 = self.single_conv(ch[1], ch[0], 5, 2)
        self.last_conv = self.single_conv(ch[0], out_channels, 5, 2)





    def __call__(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        out = self.last_conv(f3)
        return out
        



    def double_conv(self, in_channels, out_channels, kernel_size, padding):

        seq = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
                                 )

        return seq
    
    def single_conv(self, in_channels, out_channels, kernel_size, padding):

        seq = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
                                 )

        return seq
