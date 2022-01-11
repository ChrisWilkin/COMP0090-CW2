import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
import time
import sys
import os

sys.path.insert(1, '..') # add folder above to path for easy import 
#sys.path.append(os.path.dirname(__file__)[:-len('/scripts')])
import data_pipeline.DataUtils as DataUtils
import data_pipeline.DatasetClass as DatasetClass
import networks.U_net_classification_ROI as MTL

# read in Yipeng's training data
dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5','train','masks','bins','bboxes')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


#set k=4 for this network to run slightly faster
net = MTL.Unet(k=4).to(device)
net = net.double()
#print(net)

## loss and optimiser
loss_func = torch.nn.CrossEntropyLoss()
loss_func_ROI = torch.sqrt(torch.nn.MSELoss())
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# training loop for Unet
print("Training started")
training_start_time = time.time()

alpha = 1
# train for 8 epochs
epochs = 8
for epoch in range(epochs):  
    # time training and keep track of loss
    epoch_training_start_time = time.time()
    total_loss = 0.0
    first_loss = []
    for i, data in enumerate(dataloader):
        print(i)
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

        print(f' roi_labels {roi_labels}, roi ims {roi_ims}, roi_targets {roi_targets}')
        #Clear gradients
        
        optimizer.zero_grad()
        segmentation, classification, ROI = net(images)

        print(f' ROI predictions are {ROI}')
        break
    
        if i == 0:
            seg_loss = loss_func(segmentation, masks.long())
            cls_loss = loss_func(classification,bins.long())
            ROI_loss = loss_func_ROI(ROI,boxes.long())

            first_loss.append(seg_loss.item())
            first_loss.append(cls_loss.item())
            #print("seg_loss",seg_loss)
            #print("cls_loss",cls_loss)
        
        
        
        else:
            seg_loss = loss_func(segmentation, masks.long())
            cls_loss = loss_func(classification,bins.long())
            #print("seg_loss",seg_loss)
            #print("cls_loss",cls_loss)
        
            seg_loss = seg_loss*((seg_loss/first_loss[0])**alpha)
            cls_loss = cls_loss*((cls_loss/first_loss[1])**alpha)
            #print("seg_loss",seg_loss)
            #print("cls_loss",cls_loss)
            loss = seg_loss + cls_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # print out average loss for epoch
        if i % 50 == 49:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, total_loss / 50))
            total_loss = 0.0
    print('Time to train epoch = {:.2f}s'.format( time.time()-epoch_training_start_time))



print('Training done.')
print('Total training time = {:.2f}s'.format( time.time()-training_start_time))
torch.save(net.state_dict(), os.path.dirname(__file__)+'/networks/Weights/MTLk4lr001ep8v2.pt')