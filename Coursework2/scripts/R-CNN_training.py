import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import sys
import os
sys.path.insert(1, '..') # add folder above to path for easy import 
import data_pipeline.DataUtils as DataUtils
import data_pipeline.DatasetClass as DatasetClass
from networks.Half_U_net import *

BATCH = 16
LR = 0.01
MOM = 0.9
EPOCHS = 40
k = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = DatasetClass.PetSegmentationDataSet('train', 'bbox', 'bin')
dataloader = DataLoader(data, BATCH, shuffle=True)


backbone = Half_Unet(k).to(device)
backbone.out_channels = k*4
# anchor generator
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
# feature maps for ROI cropping and ROI sizes 
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
net = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
net = net.double()
criterion = optim.SGD(net.parameters(), LR, MOM) # optimizer
#loss = torch.nn.MSELoss()

lr_scheduler = torch.optim.lr_scheduler.StepLR(criterion, step_size=3, gamma=0.1) #learning rate scheduler

losses = []

for epoch in range(EPOCHS):
    for i, data in enumerate(dataloader):
        ims, labels, boxes = data.values()
        labels = labels.to(torch.int64)

        
        ims = list(image for image in ims)
        labels = labels.to(device)
        boxes = boxes.to(device)
        targets = [{'boxes': boxes[i].reshape((1,4)), 'labels': labels[i].reshape(1)} for i in range(len(labels))]

        try:
            output = net(ims, targets) # losses 

        # except value error in case bbox values = 0
        except ValueError as e:
            print(e)
            print('skipping this batch...')

        else:
            l = sum(loss for loss in output.values()) # sum loss values
            criterion.zero_grad()
            l.backward()
            criterion.step()
            losses.append(l.item())
            print(f'Batch {i}/{len(dataloader)}')
            print('Average Batch Loss: ', l.item())

            #if (i % 5) == 0:
            #    print(pred[0] * 256)
            #    print(boxes[0])
            lr_scheduler.step()
        
torch.save(net.state_dict(), 'rcnn_40epochs.pt')
