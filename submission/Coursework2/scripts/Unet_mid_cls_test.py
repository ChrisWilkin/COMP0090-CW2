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
dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5','test','masks', 'bins')
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#set k=4 for this network to run slightly faster
net = U_net.Unet(k=4).to(device)
net = net.double()

net.load_state_dict(torch.load(os.path.dirname(__file__)[:-len('/scripts')]+'/networks/Weights/Unet_Mid_clsk4lr005ep5v1.pt', map_location=device))

seg_loss_func = torch.nn.CrossEntropyLoss()
cls_loss_func = torch.nn.BCELoss()
alpha = 1
beta = 1

cls_losses = []
seg_losses = []
losses = []
seg_acc = 0
cls_acc = 0

# training loop for Unet
print("Training started")
training_start_time = time.time()

net.eval()

with torch.no_grad():

    for i, data in enumerate(dataloader):
        print(i, '/', len(dataloader))
        images, images_ID, masks, masks_ID, bins, bins_ID = data.values()
        images =  images.to(device)
        masks = masks.to(device)
        bins = bins[:, 0, None].to(device)
        
        segmentation, cls = net(images)
        seg_loss = seg_loss_func(segmentation, masks.long())
        # print(cls, torch.argmax(cls, 1)[:, None])
        cls_loss = cls_loss_func(cls.double(), bins.double())
        loss = alpha * seg_loss.double() + beta * cls_loss

        cls_losses.append(cls_loss.item())
        seg_losses.append(seg_loss.item())
        losses.append(loss.item())
        segmentation = torch.argmax(segmentation, 1)
        cls = (cls >= 0.5) * 1       
        cls_acc += torch.sum(cls == bins)
        seg_acc += torch.sum(segmentation == masks)

        img = images[0].permute(1,2,0).detach().cpu().numpy()/255
        fig, ax = plt.subplots()

        print(segmentation.shape, segmentation[0].shape)
        msk = np.ones(segmentation[0].shape)-segmentation[0].detach().cpu().numpy()
        img = img - np.repeat(msk[:, :, None], 3, axis=2)
        ax.imshow(img)

        label = "dog" if cls[0] == 1 else "cat"
        fig.text(0.25, 0.80,label,fontsize = 10,bbox ={'facecolor':'white','alpha':0.6,'pad':10})

        cls_accuracy = cls_acc / len(dataset)
        seg_accuracy = seg_acc / (len(dataset) * 256 *256)

        #fig.show()
        plt.show()

        print(f'Binary Classification Acc: {cls_accuracy:.2f}')
        print(f'Segmentation Acc: {seg_accuracy:.2f}')


    print('Testing done.')
#print('Total training time = {:.2f}s'.format( time.time()-training_start_time))
#torch.save(net.state_dict(), os.path.dirname(__file__)[:-len('/scripts')]+'/networks/Weights/Unet_Mid_clsk4lr005ep5v1.pt')