import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(__file__)[:-len('/networks')]) #Import other folders after this line

def get_test_input():
    img = Image.open("dog-cycle-car.png", 'r')
    img = np.asarray(img.resize((256,256)))         #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()
    if torch.cuda.is_available():
        img_ = img_.to('cuda')      #Convert to float
    img_ = Variable(img_)               # Convert to Variable
    return img_
    



class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchor = anchors 

    def anchors(self):
        return self.anchor

class Skip(nn.Module):
    '''
    Used as an empty layer
    '''
    def __init__(self, skip):
        super(Skip, self).__init__()
        self.skip = skip

    def forward(self, x):
        return x

    def skips(self):
        return self.skip

class Route(nn.Module):
    '''
    Used as an empty layer
    '''
    def __init__(self, layer):
        super(Route, self).__init__()
        self.layer = layer

    def forward(self, x):
        return x

    def layers(self):
        return self.layer

def generate_conv_blocks(feat1, n):
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
    count = 0
    feat2 = 2 * feat1
    blocks = nn.ModuleList() 

    block = nn.Sequential()
    block.add_module(f'conv3x3', nn.Conv2d(feat1, feat2, 3, 2, 1))
    block.add_module(f'BN', nn.BatchNorm2d(feat2))
    block.add_module(f'Leaky', nn.LeakyReLU())
    blocks.add_module(f'{count}', module=block)
    count += 1

    for i in range(n):
        block = nn.Sequential()
        block.add_module(f'conv1x1', nn.Conv2d(feat2, feat1, 1, 1, 0))
        block.add_module(f'BN_{i}', nn.BatchNorm2d(feat1))
        block.add_module(f'Leaky_{i}', nn.LeakyReLU())
        blocks.add_module(f'{count}', module=block)
        count += 1
        
        block = nn.Sequential()
        block.add_module(f'conv3x3_{i}', nn.Conv2d(feat1, feat2, 3, 1, 1))
        block.add_module(f'BN_{i }', nn.BatchNorm2d(feat2))
        block.add_module(f'Leaky_{i }', nn.LeakyReLU())
        blocks.add_module(f'{count}', module=block)
        count += 1
        
        blocks.add_module(f'{count}', Skip(-3))
        count += 1
    
    return blocks

def generate_pool_blocks(feat1, feat2, mask, n):
    '''
    feat1 - number of filters outputted by 1x1 conv
    feat2 - number of filters outputted by 3x3 conv
    mask - mask indices for detection layer
    -------------
    Returns:
        blocks: module list of sequentials
        filters: list of output filter sizes for each stage
    '''
    count = 0
    blocks = nn.ModuleList()
    anchors = [[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]]

    for i in range(3):
        block = nn.Sequential()
        block.add_module(f'conv1x1_{i}', nn.Conv2d(feat2 + feat1 if ((n==1 or n==2) and i ==0) else feat2, feat1, 1, 1, 0))
        block.add_module(f'BN_{i}', nn.BatchNorm2d(feat1))
        block.add_module(f'LeakyReLU', nn.LeakyReLU())
        blocks.add_module(f'{count}', module=block)
        count += 1

        block = nn.Sequential()
        block.add_module(f'conv3x3_{i}', nn.Conv2d(feat1, feat2, 3, 1, 1))
        block.add_module(f'BN_{i}', nn.BatchNorm2d(feat2))
        block.add_module(f'LeakyReLU', nn.LeakyReLU())
        blocks.add_module(f'{count}', module=block)
        count += 1

    block = nn.Sequential()
    block.add_module(f'conv1x1_{i}', nn.Conv2d(feat2, 21, 1, 1, 0))
    block.add_module(f'ReLU', nn.ReLU())
    blocks.add_module(f'{count}', module=block)   
    count += 1

    # YOLO layer
    anchors = [anchors[i] for i in mask]
    detection = DetectionLayer(anchors)
    blocks.add_module(f'{count}', detection)
        
    return blocks
    
def prediction_transforms(pred, inp, anchors, classes, gpu=False):
    '''
    Prediction transformation taken from 
    https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/
    '''
    batch = pred.size(0)
    stride = inp // pred.size(2)
    grids = inp // stride
    bbox = 5 + classes
    num_anchors = len(anchors)

    prediction = pred.view(batch, bbox * num_anchors, grids * grids)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch, grids * grids * num_anchors, bbox)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    grid = np.arange(grids)
    a, b = np.meshgrid(grid, grid)

    x_off = torch.FloatTensor(a).view(-1,1)
    y_off = torch.FloatTensor(b).view(-1,1)

    if gpu:
        x_off = x_off.cuda()
        y_off = y_off.cuda()
        
    
    xy_off = torch.cat((x_off, y_off), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)


    prediction[:, :, :2] += xy_off

    anchors = torch.FloatTensor(anchors)
    if gpu:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grids*grids, 1).unsqueeze(0)

    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
    prediction[: ,:, 5:5+classes] = torch.sigmoid((prediction[: ,:, 5:5+classes]))
    prediction[:, :, :4] *= stride

    return prediction

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.batch=64
        self.subdivisions=16
        self.width=256 #608
        self.height=256
        self.channels=3
        self.learning_rate=0.001
        self.mod_count = 0
        # intial convolutions
        self.features = nn.ModuleList().to('cuda')
        block = nn.Sequential()
        block.add_module('conv_0', nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False))
        block.add_module('batch_norm_0', nn.BatchNorm2d(32))
        block.add_module('leaky_0', nn.LeakyReLU())
        self.features.add_module(f'{self.mod_count}', module=block)
        self.mod_count += 1

        for i, n in enumerate([1,2,8,8,4]):
            inputs = 32 * (2**i)
            #BLOCK 
            blocks = generate_conv_blocks(inputs, n)
            for mod in blocks:
                self.features.add_module(f'{self.mod_count}', mod)
                self.mod_count += 1

            block = nn.Sequential()
            block.add_module(f'conv_{i+1}', nn.Conv2d(inputs if n != 1 else int(inputs/2), 64, 3, stride=2, padding=1, bias=False))
            block.add_module(f'batch_norm_{i+1}', nn.BatchNorm2d(364))
            block.add_module(f'leaky_{i+1}', nn.LeakyReLU())
            self.features.add_module(f'{self.mod_count}', module=block)

        ########### AvgPool - Connected - Softmax ##############

            
        for i, (mask, route) in enumerate(zip([[6,7,8], [3,4,5], [0,1,2]], [61, 36, None])):
            blocks = generate_pool_blocks(128 * (2 ** (2-i)), 256 * (2 ** (2-i)), mask, i)
            #self.features.extend(blocks)
            for mod in blocks:
                self.features.add_module(f'{self.mod_count}', mod)
                self.mod_count += 1

            if i != 2:
                feat1 = 128 * (2 ** (2-i))
                self.features.add_module(f'{self.mod_count}', Route([-4, 0]))
                self.mod_count += 1
                self.features.add_module(f'{self.mod_count}', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
                self.mod_count += 1
                self.features.add_module(f'{self.mod_count}', nn.Sequential(nn.Conv2d(feat1, int(feat1/2), 1, 1, 0),
                                                                nn.BatchNorm2d(int(feat1/2)),
                                                                nn.LeakyReLU()))
                self.mod_count += 1
                self.features.add_module(f'{self.mod_count}', Route([-1, route]))
                self.mod_count += 1

        #for name, module in self.features.named_children():
        #    print(name, module)

    def forward(self, x, gpu=False):
        outputs = {}
        mods = self.features
        write = 0
        for i, mod in enumerate(mods):
            #print(i, x.shape, type(mod))
            if isinstance(mod, nn.Sequential):
                x = self.features[i](x)

            elif isinstance(mod, nn.Upsample):
                x = mod(x)

            elif isinstance(mod, Skip):
                from_ = int(mod.skips())
                x = outputs[i-1] + outputs[i+from_]
            elif isinstance(mod, Route):
                layers = mod.layers()
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                

                if layers[1] == 0:
                    x = outputs[i + (layers[0])]
                    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
                
            elif isinstance(mod, DetectionLayer):
                anchors = mod.anchors()
                inp = 256
                classes = 2
                x = x.data
                x = prediction_transforms(x, inp, anchors, classes, gpu)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            else:
                print('Unknown module!!!')
            
            outputs[i] = x

        return detections

print('Using GPU' if torch.cuda.is_available() else 'Using CPU')

net = YOLO().to('cuda' if torch.cuda.is_available() else 'cpu')
inp = get_test_input()
pred = net(inp, torch.cuda.is_available())
print(pred.shape)