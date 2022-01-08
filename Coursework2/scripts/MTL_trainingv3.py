import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import time
import sys
import os
sys.path.append(os.path.dirname(__file__)[:-len('/scripts')])
sys.path.insert(1, '..') 
import data_pipeline.DatasetClass as DatasetClass
import networks.MTL_Components as MTL

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

K = 4
LR = 0.001
BATCH = 8
MOM = 0.9
EPOCHS = 1
CLASSES = 3 #Includes a background class = 0 for ROI
N_SEGS = 2
IN_CHANNELS = 3

#Load the data in
dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5', 'train', 'masks', 'bbox', 'bins')
dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)
valset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5', 'val', 'masks', 'bins') #Validation and Test sets do not have ROI data :(
valloader = DataLoader(valset, batch_size=BATCH, shuffle=True, num_workers=0)


#Network Components
body = MTL.Body(K, IN_CHANNELS, N_SEGS)
segment = MTL.Segmentation(K, N_SEGS, body)
roi = MTL.ROI(K, body, device)

#Losses and Criterions
seg_criterion = optim.SGD(segment.parameters(), LR, MOM)
roi_criterion = optim.SGD(roi.net.parameters(), LR, MOM)
seg_loss = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(roi_criterion, step_size=3, gamma=0.1) #learning rate scheduler

#Stored Data
seg_losses = []
roi_losses = []
total_losses = []
seg_accuracy = []
cls_accuracy = []


##################### Training Loop #######################

for epoch in range(EPOCHS):
    #Set 'per-epoch' values
    print('\nEPOCH ', epoch)
    t = time.time()
    
    for i, data in enumerate(dataloader):
        images, images_ID, masks, masks_ID, boxes, boxes_ID, bins, bins_ID = data.values()
        images = images.to(device)
        masks = masks.to(device)
        boxes = boxes.to(device)
        bins = bins.to(device)

        #Seg specific setup
        masks = masks.long()

        #ROI specific setup
        roi_labels = bins.to(torch.int64)
        roi_ims = list(image.to(device) for image in images)
        roi_targets = [{'boxes': boxes[i].reshape((1,4)), 'labels': roi_labels[i].reshape(1)} for i in range(len(roi_labels))]

        #Clear gradients
        seg_criterion.zero_grad()
        roi_criterion.zero_grad()

        #Forward pass -- First check that the networks work successfully
        try:
            seg_output = segment(images)
            roi_output = roi.forward(roi_ims, roi_targets)
        # except value error in case bbox values = 0
        except ValueError as e:
            print(e)
            print('skipping this batch...')
        else:
            #loss calc
            seg_l = seg_loss(seg_output, masks)
            roi_l = sum(loss for loss in roi_output.values()) # sum loss values

            #Backward pass
            seg_l.backward()
            roi_l.backward()

            #Optimizer step
            seg_criterion.step()
            roi_criterion.step()
            lr_scheduler.step()

            seg_losses.append(seg_l.item())
            roi_losses.append(roi_l.item())

        if (i+1) % 25 == 0:
                print(f'Batch {i}/{len(dataloader)}')
                print('Average 25 Batch Seg Loss: ', np.average(seg_losses[-25:]))
                print('Average 25 Batch ROI Loss: ', np.average(roi_losses[-25:]))
                print(f'25 Batches: {time.time() - t:.2f}s')
                t = time.time()

torch.save(body.state_dict(), f'MTLBodylr001ep{EPOCHS}.pt')
torch.save(segment.state_dict(), f'MTLSeglr001ep{EPOCHS}.pt')
torch.save(roi.net.state_dict(), f'MTLROIlr001ep{EPOCHS}.pt')

print('Saved Network Weights')
        





