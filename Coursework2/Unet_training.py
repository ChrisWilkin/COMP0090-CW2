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
import data_pipeline.DataUtils as DataUtils
import data_pipeline.DatasetClass as DatasetClass
import networks.U_net as Unet

# read in Yipeng's training data
dataset = DatasetClass.PetSegmentationDataSet('train','bin','mask',"bbox")
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

#set k=12 for this network to run slightly faster
net = Unet.Unet(k=4)
net = net.double()
#print(net)


## loss and optimiser
loss_func = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# training loop for Unet
print("Training started")
training_start_time = time.time()

# train for 10 epochs
epochs = 10
for epoch in range(epochs):  
    # time training and keep track of loss
    epoch_training_start_time = time.time()
    total_loss = 0.0

    for i, data in enumerate(dataloader):
        images, masks, bins, bbox = data.values()
    
        optimizer.zero_grad()
        output = net(images)
        new_masks = masks.view(masks.size()[0],masks.size()[2],masks.size()[3]).long()
        loss = loss_func(output, new_masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        total_loss += loss.item()

        # print out average loss for epoch
        if i % 50 == 49:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, total_loss / 50))
            total_loss = 0.0
        
    print('Time to train epoch = {:.2f}s'.format( time.time()-epoch_training_start_time))



print('Training done.')
print('Total training time = {:.2f}s'.format( time.time()-training_start_time))