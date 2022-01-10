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
import networks.MTL_Componentsv2 as MTL2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

K = 12
SEG_LR = 0.01  
ROI_LR = 0.0001
CLS_LR = 0.01
BATCH = 4
MOM = 0.9
EPOCHS = 4
CLASSES = 3 #Includes a background class = 0 for ROI
N_SEGS = 2
IN_CHANNELS = 3

PRINT = 25 # batch interval to print data at 


#Load the data in
dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5', 'train', 'masks', 'bboxes', 'bins')
sub1 = Subset(dataset, np.arange(0, 500, 1))
dataloader = DataLoader(sub1, batch_size=BATCH, shuffle=True, num_workers=0)
#valset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5', 'val', 'masks', 'bins') #Validation and Test sets do not have ROI data :(
#sub2 = Subset(valset, np.arange(0, 100, 1))
#valloader = DataLoader(sub2, batch_size=BATCH, shuffle=True, num_workers=0)


#Network Components
body = MTL2.Body(K, IN_CHANNELS, N_SEGS).to(device).double()
segment = MTL2.Segmentation(K, N_SEGS, body).to(device).double()
roi = MTL2.ROI(K, body, device)

alpha = 1 #This is the weighting for Segmentation, leave at 1 normally
beta = 0.1 #This is the weighting for ROI, try 1, 0.5, 0.25, 0.1
gamma = 0.5 #This is the weighting for Binary Classification

#Losses and Criterions
seg_criterion = optim.SGD(segment.parameters(), SEG_LR, MOM, weight_decay=0.005)
roi_criterion = optim.SGD(roi.net.parameters(), ROI_LR, MOM, weight_decay=0.005)
cls_criterion = optim.SGD(body.parameters(), CLS_LR, MOM, weight_decay=0.005)
seg_loss = torch.nn.CrossEntropyLoss()
cls_loss = torch.nn.BCELoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(roi_criterion, step_size=1, gamma=0.2) #learning rate scheduler
lr_scheduler2 = torch.optim.lr_scheduler.StepLR(seg_criterion, step_size=1, gamma=0.5) #learning rate scheduler

#Stored Data
seg_losses = []
roi_losses = []
cls_losses = []
total_losses = []
seg_accuracy = []
cls_accuracy = []


##################### Training Loop #######################
#Seems to converge around epoch 4/5

for epoch in range(EPOCHS):
    #Set 'per-epoch' values
    print('\nEPOCH ', epoch)
    t = time.time()
    t_e = time.time()
    
    for i, data in enumerate(dataloader):
        images, images_ID, masks, masks_ID, bins, bins_ID, boxes, boxes_ID = data.values()
        images = images.to(device) / 255
        masks = masks.to(device)
        boxes = boxes.to(device)
        bins = bins.to(device)
        bins = bins[:, 0] + 1 #select only cat/dog data and covert to 1/2 labels

        #Seg specific setup
        masks = masks.long()

        #ROI specific setup
        roi_labels = bins.to(torch.int64)
        roi_ims = list(image for image in images)
        roi_targets = [{'boxes': boxes[i].reshape((1,4)), 'labels': roi_labels[i].reshape(1)} for i in range(len(roi_labels))]

        #Clear gradients
        seg_criterion.zero_grad()
        roi_criterion.zero_grad()

        #Forward pass -- First check that the networks work successfully
        try:
            seg_output, bin_output = segment(images)
            roi_output = roi.forward(roi_ims, roi_targets)
        # except value error in case bbox values = 0
        except ValueError as e:
            print(e)
            print('skipping this batch...')
        else:
            #loss calc
            seg_l = alpha * seg_loss(seg_output, masks) + gamma * cls_loss(bin_output.double(), bins[:, None].double() - 1)
            roi_l = beta * sum(loss for loss in roi_output.values()) # sum loss values
            cls_l = gamma * cls_loss(bin_output.double(), bins[:, None].double() - 1)

            #Backward pass
            seg_l.backward()
            roi_l.backward()

            #Optimizer step
            seg_criterion.step()
            roi_criterion.step()

            seg_losses.append(seg_l.item())
            roi_losses.append(roi_l.item())
            cls_losses.append(cls_l.item())

        if (i+1) % PRINT == 0:
                print(f'Batch {i}/{len(dataloader)}')
                print('Average 25 Batch Seg Loss: ', np.average(seg_losses[-PRINT:]))
                print('Average 25 Batch ROI Loss: ', np.average(roi_losses[-PRINT:]))
                print('Average 25 Batch CLS Loss: ', np.average(cls_losses[-PRINT:]))
                print(f'25 Batches: {time.time() - t:.2f}s')
                t = time.time()

    lr_scheduler.step()
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
with open('MTL_segment_losses.csv', 'w') as file:
    file.write('\n'.join(str(i) for i in seg_losses))

with open('MTL_ROI_losses.csv', 'w') as file:
    file.write('\n'.join(str(i) for i in roi_losses))

with open('MTL_Cls_losses.csv', 'w') as file:
    file.write('\n'.join(str(i) for i in cls_losses))

# saving accuracy at each epoch to csv file
#with open('MTL_training_seg_accuracy.csv', 'w') as file:
#    file.write('\n'.join(str(i) for i in seg_accuracy ))

torch.save(body.state_dict(), f'YOUR-FILE-NAME-HERE.pt')
torch.save(segment.state_dict(), f'YOUR-FILE-NAME-HERE.pt')
torch.save(roi.net.state_dict(), f'YOUR-FILE-NAME-HERE.pt')

print('Saved Network Weights')
        





