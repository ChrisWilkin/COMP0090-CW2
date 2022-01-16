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


# higher level than tensor flow
class Half_Unet(nn.Module):
    def __init__(self,k=32,in_chns=3,n_segments=2):
        """Class to define half U-net architecture
        ----Args-----
        k: number of output channels from first layer, which doubles after each block
        in_chns = number of channels in input image
        n_segments = number of output channels for segmentation
        """

        super(Half_Unet,self).__init__()
        
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
        #self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

    def forward(self,x):
        x = x.double()
        #1st convolutional block
        out1 = self.conv1(x)
        maxpool1 = self.maxpool1(out1)
        #2nd convolutional block
        out2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(out2)
        #3rd convolutional block
        out3 = self.conv3(maxpool2)
        #maxpool3 = self.maxpool3(out3)
        return(out3)



#k = 4
## load a pre-trained model for classification and return
## only the features
#backbone = Half_Unet(k)
#backbone.out_channels = k*4
#
## anchor generator
#anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
#
## feature maps for ROI cropping and ROI sizes 
#roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
#
#model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
#print(model)