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



dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5','test','masks','bins')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

net = Unet.Unet(k=4).to(device)
net = net.double()
net.load_state_dict(torch.load(os.path.dirname(__file__)[:-len('/scripts')]+'/networks/Weights/Unetk4lr001ep8v2.pt', map_location=device))

net.eval()
loss_func = torch.nn.CrossEntropyLoss()
losses = []
seg_accuracy = []
cls_accuracy = []


with torch.no_grad():
    for i, data in enumerate(dataloader):
        images, images_ID, masks, masks_ID, bins, bins_ID = data.values()
        images = images.to(device)
        masks = masks.to(device)
        bins = bins[:,0].to(device)

        # calculate outputs by running images through the network
        segmentation, classification = net(images)
        seg_loss = loss_func(segmentation, masks.long())
        cls_loss = loss_func(classification,bins.long())
        #print("seg_loss",seg_loss)
        #print("cls_loss",cls_loss)
        loss = seg_loss + cls_loss
        losses.append(loss.item())

        # segmentation accuracy
        _, predicted_pixels = torch.max(segmentation.data, 1)
        _, predicted_labels = torch.max(segmentation.data, 1)

        total_pixels += masks.nelement()  # number of pixels in mask
        correct_pixels += predicted_pixels.eq(masks.data).sum().item()
        # count the number of correctly predicted images
        total_labels += bins.size(0)
        correct_labels += (predicted_labels == bins).sum().item()

# print segmentation accuracy
train_seg_accuracy = (correct_pixels / total_pixels) * 100
seg_accuracy.append(train_seg_accuracy)
print(f'Segmentation accuracy at epoch {epoch}: {round(train_seg_accuracy, 2)}')

train_cls_accuracy = (correct_labels / total_labels) * 100
cls_accuracy.append(train_cls_accuracy)
print(f'Classification accuracy at epoch {epoch}: {round(train_cls_accuracy, 2)}')

            
            

print(np.average(losses))



