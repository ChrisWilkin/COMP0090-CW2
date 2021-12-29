import torch
import torch.nn as nn
from torch.nn import modules
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

    def forward(self, x):
        return x

def generate_blocks(inputs, feat1, feat2, n):
    '''
    inputs - Number of feature maps sent into the block
    feat1 - Number of output feature2 for 1x1 Conv
    feat2 - Number of output features for 3x3 Conv
    n - Number of times this block is repeated
    '''
    blocks = nn.ModuleList()
    input_1 = inputs
    input_2 = 2 * inputs #Doubled due to skip layer concatenating two outputs
    for i in range(n):
        block = nn.Sequential()
        block.add_module(f'conv1x1_{i}', nn.Conv2d(input_1 if i == 0 else input_2, feat1, 1, 1, 0))
        block.add_module(f'BN_{i}', nn.BatchNorm2d(feat1))
        block.add_module(f'Leaky_{i}', nn.LeakyReLU())
        blocks.add_module(module=block)
        
        block = nn.Sequential()
        block.add_module(f'conv3x3_{i}', nn.Conv2d(feat1, feat2, 3, 1, 1))
        block.add_module(f'BN_{i}', nn.BatchNorm2d(feat2))
        block.add_module(f'Leaky_{i}', nn.LeakyReLU())
        blocks.add_module(module=block)

        blocks.add_module(f'Shortcut_{i}', Skip)
    
    return blocks


class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.filters = []
        
        # intial convolutions
        self.features = nn.ModuleList()
        block = nn.Sequential()
        block.add_module()

        self.features.add_module(nn.Conv2d(3, 32, 3, padding=1, bias=False))
        self.features.add_module(nn.BatchNorm2d(32))
        self.features.add_module(nn.LeakyReLU())
        

    def forward(self, x):
        return x


