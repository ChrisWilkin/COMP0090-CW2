from pickle import TRUE
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import time
import sys
import os
import argparse
sys.path.append(os.path.dirname(__file__)[:-len('/scripts')])
sys.path.insert(1, '..') 
import data_pipeline.DatasetClass as DatasetClass
import networks.MTL_Components as MTL
import networks.MTL_Componentsv2 as MTL2
import networks.MTL_Componentsv3 as MTL3


parser = argparse.ArgumentParser()
parser.add_argument('--roi', help="includes ROI in MTL", dest='inc_roi', action='store_true')
parser.add_argument('--no-roi', help="exludes ROI from MTL", dest='inc_roi', action='store_false')
parser.set_defaults(roi=True)
parser.add_argument("--seg-lr", help="segmentation learning rate", type=float, default=0.01)
parser.add_argument("--roi-lr", help="ROI learning rate", type=float, default=0.0001)
parser.add_argument("--mom",  help="ROI and segmentation optimizer momentum", type=float, default=0.9)
parser.add_argument("--branch", help="branching location - middle or end", default="middle")
parser.add_argument("--alpha", help="weighting for segmentation losses", type=float, default=1.0)
parser.add_argument("--beta", help="weighting for ROI losses", type=float, default=1.0)
parser.add_argument("--gamma", help="weighting for classification losses", type=float, default=1.0)
parser.add_argument("--step", help="learning rate step size", type=int, default=2)
parser.add_argument("--seg-losses", help="csv filename for storing segmentation losses", type=str, default="MTL_Seg_losses.csv")
parser.add_argument("--roi-losses", help="csv filename for storing ROI losses", type=str, default="MTL_ROI_losses.csv")
parser.add_argument("--cls-losses", help="csv filename for storing classification losses", type=str, default="MTL_Cls_losses.csv")
parser.add_argument("--body", help="filename for storing state dict of MTL body after training", type=str, default="MTL_Body.pt")
parser.add_argument("--seg", help="filename for storing state dict of segmentation after training", type=str, default="MTL_Seg.pt")
parser.add_argument("--r", help="filename for storing state dict of ROI after training", type=str, default="MTL_ROI.pt")
parser.add_argument("--epochs", help="number of training epochs", type=int, default=10)
parser.add_argument('--task', help="does not randomly choose which task to update each batch", dest='rand_weights', action='store_false')
parser.add_argument('--rand-task', help="randomly choose which task to update each batch", dest='rand_weights', action='store_true')
parser.set_defaults(task=False)
args = parser.parse_args()

inc_roi = args.inc_roi
if inc_roi:
    ROI_LR = args.roi_lr  #ROI
    roi_filename = args.r
    roilosses_filename = args.roi_losses
else:
    print('Excluding ROI from MTL network.')

if args.branch == 'middle' or args.branch == 'end':
    branch = args.branch
else:
    raise ValueError('incorrect branching specified')

SEG_LR = args.seg_lr
MOM = args.mom
alpha = args.alpha # This is the weighting for Segmentation losses
beta = args.beta # This is the weighting for ROI losses (1, 0.5, 0.25, 0.1)  #ROI
gamma = args.gamma    # THis is the weighting for Binary Cls (1, 0.01)
step = args.step
seglosses_filename = args.seg_losses
clslosses_filename = args.cls_losses
body_filename = args.body
seg_filename = args.seg
EPOCHS = args.epochs

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

K = 12
BATCH = 6
CLASSES = 3 #Includes a background class = 0 for ROI
N_SEGS = 2
IN_CHANNELS = 3

PRINT = 50 # batch interval to print data at 


#Load the data in
dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5', 'train', 'masks', 'bboxes', 'bins')
sub1 = Subset(dataset, np.arange(0, 500, 1))
dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)
#valset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5', 'val', 'masks', 'bins') #Validation and Test sets do not have ROI data :(
#sub2 = Subset(valset, np.arange(0, 100, 1))
#valloader = DataLoader(sub2, batch_size=BATCH, shuffle=True, num_workers=0)


#Network Components
if branch == 'middle': # early branching
    body = MTL3.Body(K, IN_CHANNELS, N_SEGS).to(device).double()
    segment = MTL3.Segmentation(K, N_SEGS, body).to(device).double()
    if inc_roi:
        roi = MTL3.ROI(K, body, device)   #ROI
else: # late branching 
    body = MTL2.Body(K, IN_CHANNELS, N_SEGS).to(device).double()
    segment = MTL2.Segmentation(K, N_SEGS, body).to(device).double()
    if inc_roi:
        roi = MTL2.ROI(K, body, device)   #ROI   

#Losses and Criterions
seg_criterion = optim.SGD(segment.parameters(), SEG_LR, MOM, weight_decay=0.005)
if inc_roi:
    roi_criterion = optim.SGD(roi.net.parameters(), ROI_LR, MOM, weight_decay=0.005)   #ROI
#cls_criterion = optim.SGD(body.parameters(), CLS_LR, MOM, weight_decay=0.005)
seg_loss = torch.nn.CrossEntropyLoss()
cls_loss = torch.nn.BCELoss()
if inc_roi:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(roi_criterion, step_size=step, gamma=0.1) #learning rate scheduler   #ROI
lr_scheduler2 = torch.optim.lr_scheduler.StepLR(seg_criterion, step_size=step, gamma=0.2) #learning rate scheduler

#Stored Data
seg_losses = []
if inc_roi:
    roi_losses = []    #ROI
cls_losses = []
total_losses = []
seg_accuracy = []
cls_accuracy = []

RANDOM_WEIGHTS = args.rand_weights  #Trigger random optimiser steps (randomly choose which task to update each  batch)
ROI_UPDATE = inc_roi   #ROI
SEG_UPDATE = True


##################### Training Loop #######################
#Seems to converge around epoch 4/5

for epoch in range(EPOCHS):
    #Set 'per-epoch' values
    print('\nEPOCH ', epoch)
    t = time.time()
    t_e = time.time()
    
    for i, data in enumerate(dataloader):
        ROI_UPDATE = inc_roi   #ROI
        SEG_UPDATE = True
        if RANDOM_WEIGHTS:
            i = np.random.randint(2)
            if i == 1:
                ROI_UPDATE = False
            elif i == 0:
                SEG_UPDATE = False

        images, images_ID, masks, masks_ID, bins, bins_ID, boxes, boxes_ID = data.values()
        images = images.to(device) / 255
        masks = masks.to(device)
        boxes = boxes.to(device)
        bins = bins.to(device)
        bins = bins[:, 0] + 1 #select only cat/dog data and covert to 1/2 labels

        #Seg specific setup
        masks = masks.long()

        if inc_roi:
            #ROI specific setup
            roi_labels = bins.to(torch.int64)   #ROI
            roi_ims = list(image for image in images)   #ROI
            roi_targets = [{'boxes': boxes[i].reshape((1,4)), 'labels': roi_labels[i].reshape(1)} for i in range(len(roi_labels))]   #ROI

        #Clear gradients
        seg_criterion.zero_grad()
        if inc_roi:
            roi_criterion.zero_grad()  #ROI

        #Forward pass -- First check that the networks work successfully
        try:
            seg_output, bins_output = segment(images)
            if inc_roi:
                roi_output = roi.forward(roi_ims, roi_targets)   #ROI
        # except value error in case bbox values = 0
        except ValueError as e:
            print(e)
            print('skipping this batch...')
        else:
            #loss calc
            seg_l = alpha * seg_loss(seg_output, masks) 
            if inc_roi:
                roi_l = beta * sum(loss for loss in roi_output.values()) # sum loss values  #ROI
            cls_l = gamma * cls_loss(bins_output.double(), bins[:, None].double() - 1)
            l = seg_l + cls_l

            #Backward pass
            seg_l.backward()
            if inc_roi:
                roi_l.backward()   #ROI

            #Optimizer step
            if SEG_UPDATE:
                seg_criterion.step()
            if ROI_UPDATE:   #ROI
                roi_criterion.step()   #ROI

            seg_losses.append(seg_l.item())
            if inc_roi:
                roi_losses.append(roi_l.item())   #ROI
            cls_losses.append(cls_l.item())

        if (i+1) % PRINT == 0:
                print(f'Batch {i}/{len(dataloader)}')
                print(f'Average {PRINT} Batch Seg Loss: ', np.average(seg_losses[-PRINT:]))
                if inc_roi:
                    print(f'Average {PRINT} Batch ROI Loss: ', np.average(roi_losses[-PRINT:]))   #ROI
                print(f'Average {PRINT} Batch CLS Loss: ', np.average(cls_losses[-PRINT:]))
                print(f'{PRINT} Batches: {time.time() - t:.2f}s')
                t = time.time()

    if inc_roi:
        lr_scheduler.step()   #ROI
    lr_scheduler2.step()

    #total_pixels = 0
    #correct_pixels = 0
#
#    #with torch.no_grad():
#    #    segment.train(False)
#    #    body.train(False)
#    #    for i, data in enumerate(valloader):
#    #        images, images_ID, masks, masks_ID, bins, bins_ID = data.values()
#    #        images = images.to(device)
#    #        masks = masks.to(device)
#
#    #        # calculate outputs by running images through the network
#    #        output = segment(images)
#
#    #        # segmentation accuracy
#    #        _, predicted = torch.max(output.data, 1)
#    #        total_pixels += masks.nelement()  # number of pixels in mask
#    #        correct_pixels += predicted.eq(masks.data).sum().item()
#
#    ##print(segmentation_accuracy)
#    #train_accuracy = (correct_pixels / total_pixels) * 100
#    #seg_accuracy.append(train_accuracy)
#    #print(f'Segmentation accuracy at epoch {epoch}: {round(train_accuracy, 2)}')
#    #print(f'Time for Epoch: {time.time() - t_e:.2f}s')
#    #
#    #segment.train(True)
    #body.train(True)

# saving the loss at each epoch to csv file
with open(seglosses_filename , 'w') as file:
    file.write('\n'.join(str(i) for i in seg_losses))

if inc_roi:
    with open(roilosses_filename , 'w') as file:   #ROI
        file.write('\n'.join(str(i) for i in roi_losses))   #ROI

with open(clslosses_filename , 'w') as file:
    file.write('\n'.join(str(i) for i in cls_losses))

# saving accuracy at each epoch to csv file
#with open('MTL_training_seg_accuracy.csv', 'w') as file:
#    file.write('\n'.join(str(i) for i in seg_accuracy ))

#Make sure that all these files are named clearly and uniquely! Not something like 'MTLv3training.pt'!!!
#This will save to the root folder by default.
torch.save(body.state_dict(), body_filename)   #Network weights for Body 
torch.save(segment.state_dict(), seg_filename)    #Netowkr weights for segmentation
if inc_roi:
    torch.save(roi.net.state_dict(), roi_filename)   #Network weights for ROI  #ROI

print('Saved Network Weights')
        





