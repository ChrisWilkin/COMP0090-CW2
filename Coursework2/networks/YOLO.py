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
import numpy as np
sys.path.append(os.path.dirname(__file__)[:-len('/networks')]) #Import other folders after this line


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors 

class Skip(nn.Module):
    '''
    Used as an empty layer
    '''
    def __init__(self):
        super(Skip, self).__init__()

    def forward(self, x):
        return x

def generate_conv_blocks(feat1, feat2, n, ind):
    '''
    inputs - Number of feature maps sent into the block
    feat1 - Number of output feature2 for 1x1 Conv
    feat2 - Number of output features for 3x3 Conv
    n - Number of times this block is repeated
    -------------
    Returns:
        blocks: module list of sequentials
        filters: list of output filter sizes for each stage
    '''
    blocks = nn.ModuleList()
    input_1 = feat2
    input_2 = 2 * feat2 #Doubled due to skip layer concatenating two outputs
    filters = []
    for i in range(n):
        block = nn.Sequential()
        block.add_module(f'conv1x1_{i + ind}', nn.Conv2d(input_1 if i == 0 else input_2, feat1, 1, 1, 0))
        block.add_module(f'BN_{i + ind}', nn.BatchNorm2d(feat1))
        block.add_module(f'Leaky_{i + ind}', nn.LeakyReLU())
        blocks.add_module('block', module=block)
        filters.append(feat1)
        
        block = nn.Sequential()
        block.add_module(f'conv3x3_{i + ind}', nn.Conv2d(feat1, feat2, 3, 1, 1))
        block.add_module(f'BN_{i + ind}', nn.BatchNorm2d(feat2))
        block.add_module(f'Leaky_{i + ind}', nn.LeakyReLU())
        blocks.add_module('block', module=block)
        filters.append(feat2)
        
        blocks.add_module(f'Shortcut_{i + ind}', Skip())
        filters.append(2 * feat2)
    
    return blocks, np.array(filters)

def generate_pool_blocks(feat1, feat2, mask):
    '''
    feat1 - number of filters outputted by 1x1 conv
    feat2 - number of filters outputted by 3x3 conv
    mask - mask indices for detection layer
    '''
    blocks = nn.ModuleList()
    filters = []
    anchors = [[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]]

    for i in range(3):
        block = nn.Sequential()
        block.add_module(f'conv1x1_{i}', nn.Conv2d(feat2 if i != 0 else 2 * feat2, feat1, 1, 1, 0))
        block.add_module(f'BN_{i}', nn.BatchNorm2d(feat1))
        block.add_module(f'LeakyReLU', nn.LeakyReLU())
        blocks.add_module('block', module=block)
        filters.append(feat1)

        block = nn.Sequential()
        block.add_module(f'conv3x3_{i}', nn.Conv2d(feat1, feat2, 3, 1, 1))
        block.add_module(f'BN_{i}', nn.BatchNorm2d(feat2))
        block.add_module(f'LeakyReLU', nn.LeakyReLU())
        blocks.add_module('block', module=block)
        filters.append(feat2)

    block = nn.Sequential()
    block.add_module(f'conv1x1_{i}', nn.Conv2d(feat2, int(feat1 / 2) - 1, 1, 1, 0))
    block.add_module(f'ReLU', nn.ReLU())
    blocks.add_module('block', module=block)   
    filters.append((feat1 / 2) - 1)


    # YOLO layer
    anchors = [anchors[i] for i in mask]
    detection = DetectionLayer(anchors)
    blocks.add_module(f'Detection_{i}', detection)
        
    return blocks, filters
    

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.filters = np.array([0])
        
        # intial convolutions
        self.features = nn.ModuleList()
        block = nn.Sequential()
        block.add_module('conv_0', nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False))
        block.add_module('batch_norm_0', nn.BatchNorm2d(32))
        block.add_module('leaky_0', nn.LeakyReLU())
        self.features.add_module('block', module=block)

        for i, n in enumerate([1,2,8,8,4]):
            inputs = 64 * (2**n)
            #BLOCK 
            blocks, f = generate_conv_blocks(inputs, 64, n, i)
            self.filters = np.concatenate((self.filters, f))
            self.features.extend(blocks)

            block = nn.Sequential()
            block.add_module(f'conv_{i+1}', nn.Conv2d(inputs if n != 1 else int(inputs/2), 64, 3, stride=2, padding=1, bias=False))
            block.add_module(f'batch_norm_{i+1}', nn.BatchNorm2d(364))
            block.add_module(f'leaky_{i+1}', nn.LeakyReLU())
            self.features.add_module('block', module=block)

        ########### AvgPool - Connected - Softmax ##############

            
        for i, mask in enumerate([[6,7,8], [3,4,5], [0,1,2]]):

            blocks, f = generate_pool_blocks(128 * (2 ** (2-i)), 256 * (2 ** (2-i)), mask)
            self.filters = np.concatenate((self.features, f))
            self.features.extend(blocks)

            if i != 2:
                feat1 = 128 * (2 ** (2-i)) - 1
                self.features.add_module(f'Route_{i}', Skip())
                self.features.add_module(f'Upsample_{i}', nn.Upsample(scale_factor=2, mode='bilinear'))
                self.features.add_module(f'Conv1x1_{i}', nn.Conv2d(int(feat1 / 2) - 1, int(feat1/2), 1, 1, 0))
                self.features.add_module(f'Route_{i}', Skip())
            
    def forward(self, x):
        
        return x

net = YOLO()
