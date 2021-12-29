import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
import sys
import os
sys.path.append(os.path.dirname(__file__)[:-len('/networks')]) #Import other folders after this line

class Skip(nn.Module):
    def __init__(self):
        super(Skip, self).__init__()

class ConvBlock(nn.Module):
    def __init__(self, inputs, feat1, feat2, n):
        '''
        inputs - Number of feature maps sent into the block
        feat1 - Number of output feature2 for 1x1 Conv
        feat2 - Number of output features for 3x3 Conv
        n - Number of times this block is repeated
        '''
        super(ConvBlock, self).__init__()
        self.blocks = nn.ModuleList()
        self.filters = []
        self.input_1 = inputs
        self.input_2 = 2 * inputs #Doubled due to skip layer concatenating two outputs
        for i in range(n):
            block = nn.Sequential()
            block.add_module(f'conv1x1_{i}', nn.Conv2d(self.input_1 if i == 0 else self.input_2, feat1, 1, 1, 0))
            block.add_module(f'conv3x3_{i}', nn.Conv2d(feat1, feat2, 3, 1, 1))
            block.add_module(f'Skip_{i}', Skip)
            self.blocks.add_module(f'Block_{i}', block)

    def forward(self, x, feature_cache):
        
        return x

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.filters = []



    def forward(self, x):
        return x


