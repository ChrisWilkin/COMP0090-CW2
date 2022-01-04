import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
import sys
import os
sys.path.insert(1, '..') # add folder above to path for easy import 


class SimpleRoI(nn.Module):
    def __init__(self):
        super(SimpleRoI, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), 
                                nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, 3, 2, 1), 
                                nn.ReLU())

        self.blocks = nn.ModuleList([self._block(32), self._block(64), self._block(128), self._block(256)])

        self.linear = nn.Linear(8*8*512, 4)

    def forward(self, x):
        outs = []
        out = self.conv1(x)
        out = self.conv2(out)
        
        for block in self.blocks:
            for i, seq in enumerate(block):
                out = seq(out)
                outs.append(out)
                if i == 2:
                    out = torch.cat((outs[-3], outs[-1]), dim=1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        out = torch.sigmoid(out)
        return out
        
    def _block(self, filters):
        a = nn.Sequential(nn.Conv2d(filters, filters, 3, 1, 1),
                            nn.BatchNorm2d(filters), 
                            nn.LeakyReLU())
        b = nn.Sequential(nn.Conv2d(filters, 2 * filters, 3, 1, 1),
                            nn.BatchNorm2d(2 * filters), 
                            nn.LeakyReLU())
        c = nn.Sequential(nn.Conv2d(2 * filters, filters, 3, 1, 1),
                            nn.BatchNorm2d(filters), 
                            nn.LeakyReLU())
        d = nn.Sequential(nn.Conv2d(2 * filters, 2 * filters, 3, 2, 1),
                            nn.BatchNorm2d(2 * filters), 
                            nn.LeakyReLU())

        return nn.ModuleList([a, b, c, d])

