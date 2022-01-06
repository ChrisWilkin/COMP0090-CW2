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
import data_pipeline.DataUtils as DataUtils
import data_pipeline.DatasetClass as DatasetClass
import networks.U_net as Unet

# read in Yipeng's training data
dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5','train','masks')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
valset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5','val','masks')
valloader = DataLoader(valset, batch_size=10, shuffle=True, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#set k=4 for this network to run slightly faster
net = Unet.Unet(k=4).to(device)
net = net.double()
#print(net)


## loss and optimiser
loss_func = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# training loop for Unet
print("Training started")
training_start_time = time.time()

# train for 8 epochs
epochs = 8
losses = []
for epoch in range(epochs):  
    # time training and keep track of loss
    epoch_training_start_time = time.time()
    total_loss = 0.0

    

    for i, data in enumerate(dataloader):
        images, images_ID, masks, masks_ID = data.values()
        images =  images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        output = net(images)
        loss = loss_func(output, masks.long())
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
    
    

    # with out the gradient calculate testing accuracy after each epoch of training
    #segmentation accuracy
    total_pixels = 0
    correct_pixels = 0
    
    with torch.no_grad():
        for i, data in enumerate(valloader):
            images, images_ID, masks, masks_ID = data.values()
            images =  images.to(device)
            masks = masks.to(device)

            # calculate outputs by running images through the network
            output = net(images)

            # segmentation accuracy
            _, predicted = torch.max(output.data, 1)
            total_pixels += masks.nelement() # number of pixels in mask
            correct_pixels += predicted.eq(masks.data).sum().item()
    

    # print segmentation accuracy
    train_accuracy = (correct_pixels/total_pixels)*100
    print(f'Segmentation accuracy at epoch {epoch}: {round(train_accuracy,2)}')


print('Training done.')
print('Total training time = {:.2f}s'.format( time.time()-training_start_time))
torch.save(net.state_dict(), os.path.dirname(__file__)+'/networks/Weights/Unetk4lr001ep8v2.pt')