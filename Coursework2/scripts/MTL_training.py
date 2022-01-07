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
import networks.U_net_classification as MTL

# read in Yipeng's training data
dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5','train','masks','bins')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
valset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5', 'val', 'masks','bins')
valloader = DataLoader(valset, batch_size=10, shuffle=True, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


#set k=4 for this network to run slightly faster
net = MTL.Unet(k=4).to(device)
net = net.double()
#print(net)


## loss and optimiser
loss_func = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# training loop for Unet
print("Training started")
training_start_time = time.time()

#alpha = 1
# train for 8 epochs
epochs = 8
losses = []
seg_accuracy = []
cls_accuracy = []

for epoch in range(epochs):  
    # time training and keep track of loss
    
    epoch_training_start_time = time.time()
    total_loss = 0.0
    for i, data in enumerate(dataloader):
        images, images_ID, masks, masks_ID, bins, bins_ID = data.values()
        images = images.to(device)
        masks = masks.to(device)
        bins = bins[:,0].to(device)
        
        optimizer.zero_grad()
        
        segmentation, classification = net(images)
        
        seg_loss = loss_func(segmentation, masks.long())
        cls_loss = loss_func(classification,bins.long())
        #print("seg_loss",seg_loss)
        #print("cls_loss",cls_loss)
        loss = seg_loss + cls_loss
        #print(loss)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        losses.append(loss.item())

        # print out average loss for epoch
        if i % 50 == 49:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, total_loss / 50))
            total_loss = 0.0
    print('Time to train epoch = {:.2f}s'.format( time.time()-epoch_training_start_time))

    total_pixels = 0
    correct_pixels = 0
    total_labels = 0
    correct_labels = 0

    with torch.no_grad():
        for i, data in enumerate(valloader):
            images, images_ID, masks, masks_ID, bins, bins_ID = data.values()
            images = images.to(device)
            masks = masks.to(device)
            bins = bins[:,0].to(device)

            # calculate outputs by running images through the network
            segmentation, classification = net(images)

            # segmentation accuracy
            _, predicted_pixels = torch.max(segmentation.data, 1)
            _, predicted_labels = torch.max(classification.data, 1)
            
            total_pixels += masks.nelement()  # number of pixels in mask
            correct_pixels += predicted_pixels.eq(masks.data).sum().item()
            # count the number of correctly predicted images
            total_labels += bins.size(0)
            correct_labels += (predicted_labels == bins).sum().item()

    # print segmentation accuracy
    train_seg_accuracy = (correct_pixels / total_pixels) * 100
    seg_accuracy.append(train_seg_accuracy)
    print(f'Segmentation accuracy at epoch {epoch+1}: {round(train_seg_accuracy, 2)}')
    
    train_cls_accuracy = (correct_labels / total_labels) * 100
    cls_accuracy.append(train_cls_accuracy)
    print(f'Classification accuracy at epoch {epoch+1}: {round(train_cls_accuracy, 2)}')

print('Training done.')
print('Total training time = {:.2f}s'.format( time.time()-training_start_time))
torch.save(net.state_dict(), os.path.dirname(__file__)+'/networks/Weights/MTLk4lr001ep8v2.pt')