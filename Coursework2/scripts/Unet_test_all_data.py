import numpy as np
import torch
import time
import sys
import os
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.dirname(__file__)[:-len('/scripts')])
sys.path.insert(1, '..') # add folder above to path for easy import 
import data_pipeline.DataUtils as DataUtils
import data_pipeline.DatasetClass as DatasetClass
import networks.U_net as Unet

#load data
dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5','test','masks')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

net = Unet.Unet(k=4).to(device)
net = net.double()
net.load_state_dict(torch.load(os.path.dirname(__file__)[:-len('/scripts')]+'/networks/Weights/Unetk4lr0005ep10v1.pt', map_location=device))

loss_func = torch.nn.CrossEntropyLoss()

losses = []
imgs, msks = [], []
net.eval()
total_pixels = 0
correct_pixels = 0

IOU_testset = []
IOU_list = []

with torch.no_grad():
    for i, data in enumerate(dataloader):
        images, images_ID, masks, masks_ID = data.values()
        images =  images.to(device)
        masks = masks.to(device)

        output = net(images)
        loss = loss_func(output, masks.long())
        losses.append(loss.item())
        
        _, predicted = torch.max(output.data, 1)
        total_pixels += masks.nelement() # number of pixels in mask
        correct_pixels += predicted.eq(masks.data).sum().item()
        
        # segmentation IOU for each image
        batch_IOU = []
        for i in range(len(predicted)):
            intersection = torch.sum(torch.logical_and(predicted[i],masks[i]))
            union = torch.sum(torch.logical_or(predicted[i],masks[i]))
            IOU = intersection/union
            batch_IOU.append(IOU.item())
            IOU_testset.append(IOU.item())
        IOU_batch = np.average(batch_IOU) * 100
        IOU_list.append(IOU_batch)
        print("Average batch IOU:", round(IOU_batch,2))
        if i == 10:
            imgs = images[0]
            msks = output[0]
        
test_accuracy = (correct_pixels/total_pixels)*100
IOU_test = np.average(IOU_testset) * 100


print(f'Segmentation accuracy on test set: {round(test_accuracy,2)}')   
 
with open('Unet_alldata_test_Accuracy.csv', 'w') as file:
    file.write(f'{test_accuracy}')
        
with open('u_net_test_losses.csv', 'w') as file:
    file.write('\n'.join(str(i) for i in losses))

            
with open('UNet_IOU_ALL_DATA.csv', 'w') as file:
    file.write('\n'.join(str(i) for i in IOU_list))
    file.write('\n'+f'{IOU_test}')            

print(np.average(losses))

msks = msks.detach().cpu()

mask = np.argmax(msks, 0)
image = imgs.detach().cpu()
print(image.shape)

DataUtils.visualise_masks(image, mask)
