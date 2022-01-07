import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(1, '..') # add folder above to path for easy import 
sys.path.append(os.path.dirname(__file__)[:-len('/scripts')])
import data_pipeline.DataUtils as DataUtils
import data_pipeline.DatasetClass as DatasetClass
import networks.U_net_mid_classification as U_net

# read in Chris's training data
dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5','train','masks', 'bins')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#set k=4 for this network to run slightly faster
net = U_net.Unet(k=4).to(device)
net = net.double()

seg_loss_func = torch.nn.CrossEntropyLoss()
cls_loss_func = torch.nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
alpha = 1
beta = 1

cls_losses = []
seg_losses = []
losses = []

# training loop for Unet
print("Training started")
training_start_time = time.time()

# train for 8 epochs
epochs = 10
for epoch in range(epochs):  
    # time training and keep track of loss
    epoch_training_start_time = time.time()
    total_loss = 0.0

    for i, data in enumerate(dataloader):
        images, images_ID, masks, masks_ID, bins, bins_ID = data.values()
        images =  images.to(device)
        masks = masks.to(device)
        bins = bins[:, 0, None].to(device)
        
        optimizer.zero_grad()
        segmentation, cls = net(images)
        seg_loss = seg_loss_func(segmentation, masks.long())
       # print(cls, torch.argmax(cls, 1)[:, None])
        cls_loss = cls_loss_func(cls.double(), bins.double())
        loss = alpha * seg_loss.double() + beta * cls_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        cls_losses.append(cls_loss.item())
        seg_losses.append(seg_loss.item())
        losses.append(loss.item())

        # print out average loss for epoch
        if i % 50 == 49:    # print every 50 mini-batches
            print(f'Pred: {cls}, Loss: {cls_loss}')
            print(f'True: {bins}')
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, total_loss / 50))
            total_loss = 0.0
    print('Time to train epoch = {:.2f}s'.format( time.time()-epoch_training_start_time))

plt.plot(losses)
plt.plot(cls_losses)
plt.plot(seg_losses)
plt.savefig('Mid-point Classification Losses.jpg')


print('Training done.')
print('Total training time = {:.2f}s'.format( time.time()-training_start_time))
torch.save(net.state_dict(), os.path.dirname(__file__)[:-len('/scripts')]+'/networks/Weights/Unet_Mid_clsk4lr005ep5v1.pt')