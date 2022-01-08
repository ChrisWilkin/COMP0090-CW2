import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
import torchvision 
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import sys
import os
sys.path.append(os.path.dirname(__file__)[:-len('/networks')])


# higher level than tensor flow
class Body(nn.Module):
    def __init__(self,k=32,in_chns=3,n_segments=2):
        """Class to define U-net architecture
        ----Args-----
        k: number of output channels from first layer, which doubles after each block
        in_chns = number of channels in input image
        n_segments = number of output channels for segmentation
        """

        super(Body,self).__init__()
        
        # 1st convolutional block
        # use padding = 1 to keep image the same size
        # use batch normalisation and Relu at each step
        # increase channels (3 -> k)
        # im size = 256x256
        self.conv1 = nn.Sequential(BatchNorm2d(in_chns),ReLU(inplace=True),Conv2d(in_chns,k,kernel_size=3,stride=1,padding=1)
                                    ,BatchNorm2d(k),ReLU(inplace=True),Conv2d(k,k,kernel_size=3,stride=1,padding=1))
        self.maxpool1= nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        # im size = 128x128
        # 2nd conv block, double output channels (k -> 2k)
        self.conv2 = nn.Sequential(BatchNorm2d(k),ReLU(inplace=True),Conv2d(k,2*k,kernel_size=3,stride=1,padding=1)
                                    ,BatchNorm2d(2*k),ReLU(inplace=True),Conv2d(2*k,2*k,kernel_size=3,stride=1,padding=1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        # im size = 64 x 64
        # 3rd conv block, double output channels (2k -> 4k)
        self.conv3 = nn.Sequential(BatchNorm2d(2*k),ReLU(inplace=True),Conv2d(2*k,4*k,kernel_size=3,stride=1,padding=1)
                                    ,BatchNorm2d(4*k),ReLU(inplace=True),Conv2d(4*k,4*k,kernel_size=3,stride=1,padding=1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        # Bottom of U-net
        # im size = 32 x 32
        # 4th conv block, half output channels (8k -> 4k)
        self.conv4 = nn.Sequential(BatchNorm2d(4*k),ReLU(inplace=True),Conv2d(4*k,8*k,kernel_size=3,stride=1,padding=1)
                                    ,BatchNorm2d(8*k),ReLU(inplace=True),Conv2d(8*k,8*k,kernel_size=3,stride=1,padding=1))
        self.upconv1 = nn.Sequential(nn.Upsample(scale_factor=(2,2), mode='bilinear'), Conv2d(8*k,4*k,kernel_size=3,stride=1,padding=1))

        # im size = 64 x 64
        # 5th conv block, concat output from 4th and 3rd block to use as inputs. 
        # Input channels = 4k + 4k = 8kd
        # output channels reduced by 4 vs input (8k -> 2k)
        self.conv5 = nn.Sequential(BatchNorm2d(8*k),ReLU(inplace=True),Conv2d(8*k,4*k,kernel_size=3,stride=1,padding=1)
                                    ,BatchNorm2d(4*k),ReLU(inplace=True),Conv2d(4*k,4*k,kernel_size=3,stride=1,padding=1))
        self.upconv2 = nn.Sequential(nn.Upsample(scale_factor=(2,2), mode='bilinear'), Conv2d(4*k,2*k,kernel_size=3,stride=1,padding=1))

        #BRANCHING POINT

        self.out1 = None
        self.out2 = None


    def forward(self,x):
        x = x.double()
        #1st convolutional block
        out1 = self.conv1(x)
        self.out1 = out1
        maxpool1 = self.maxpool1(out1)
        #2nd convolutional block
        out2 = self.conv2(maxpool1)
        self.out2 = out2
        maxpool2 = self.maxpool2(out2)
        #3rd convolutional block
        out3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(out3)
        #4th convolutional block
        #BOTTOM of U-net
        out4 = self.conv4(maxpool3)
        upconv1 = self.upconv1(out4)
        #5th convolutional block 
        #concat output from 4th and 3rd block as inputs
        out5 = self.conv5(torch.cat([upconv1,out3],1))
        upconv2 = self.upconv2(out5)
    
        return upconv2

class Segmentation(nn.Module):
    def __init__(self, k, n_segments, body: Body):
        super(Segmentation, self).__init__()

        self.body = body
        self.out1 = self.body.out1
        self.out2 = self.body.out2
        #CONTINUATION OF UNET STEM
        # im size = 128 x 128
        # 6th conv block, concat outputs from 5th and 2nd blocks to use as inputs.
        # Input channels = 2k + 2k = 4k
        # output channels reduced by 4 vs input (4k -> k)
        self.conv6 = nn.Sequential(BatchNorm2d(4*k),ReLU(inplace=True),Conv2d(4*k,2*k,kernel_size=3,stride=1,padding=1)
                                    ,BatchNorm2d(2*k),ReLU(inplace=True),Conv2d(2*k,2*k,kernel_size=3,stride=1,padding=1))
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=(2,2), mode='bilinear'), Conv2d(2*k,k,kernel_size=3,stride=1,padding=1))

        # im size = 256 x 256
        # 7th and final  conv block, concat input with output from 1st block. 
        # Input = k + k = 2k
        self.conv7 = nn.Sequential(BatchNorm2d(2*k),ReLU(inplace=True),Conv2d(2*k,k,kernel_size=3,stride=1,padding=1)
                                    ,BatchNorm2d(k),ReLU(inplace=True),Conv2d(k,k,kernel_size=3,stride=1,padding=1)
        # final prediction layer for segmentation
                                   ,Conv2d(k,n_segments,kernel_size=3,stride=1,padding=1))

    def forward(self, x):
        upconv2 = self.body(x)
        #6th convolutional block 
        #concat output from 5th and 2nd blocks as inputs
        out6 = self.conv6(torch.cat([upconv2, self.out2],1))
        upconv3 = self.upconv3(out6)
        #7th and final block
        #concat input from 6th and 1st block
        out7 = self.conv7(torch.cat([upconv3, self.out1],1))

        return out7

class ROI():
    def __init__(self, k, body: Body, device='cpu'):
        self.backbone = body.to(device)
        self.backbone.out_channels = k*4

        self.anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        # feature maps for ROI cropping and ROI sizes 
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        self.net = FasterRCNN(self.backbone, num_classes=3, rpn_anchor_generator=self.anchor_generator, box_roi_pool=self.roi_pooler).to(device)
        self.net = self.net.double()

    def forward(self, images, targets: dict):
        assert 'boxes' in targets.keys
        assert 'labels' in targets.keys

        return self.net(images, targets)

    def train(self):
        '''
        Sets the model to train mode.
        '''
        self.net.train()
        return
    
    def eval(self):
        '''
        Sets the model to evaluation mode.
        '''
        self.net.eval()
        return



