import numpy as np
import torch
import time
import sys
import os
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.dirname(__file__)[:-len('/scripts')])
import data_pipeline.DataUtils as DataUtils
import data_pipeline.DatasetClass as DatasetClass
import networks.U_net as Unet

#load data
dataset = DatasetClass.PetSegmentationDataSet('test','bin','mask',"bbox")
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

net = Unet.Unet(k=4).to(device)
net = net.double()
net.load_state_dict(torch.load(os.path.dirname(__file__)[:-len('/scripts')]+'/networks/Weights/Unetk4lr001ep8v1.pt', map_location=device))

loss_func = torch.nn.CrossEntropyLoss()

losses = []
imgs, msks = [], []
net.eval()

# segmentation accuracy
total_pixels = 0
correct_pixels = 0

for i, data in enumerate(dataloader):
    images, masks, bins, bbox = data.values()
    images =  images.to(device)
    masks = masks.to(device)
    bins = bins.to(device)
    bbox = bbox.to(device)

    output = net(images)
    new_masks = masks.view(masks.size()[0],masks.size()[2],masks.size()[3]).long()
    loss = loss_func(output, new_masks)
    losses.append(loss.item())
    if i == 10:
        imgs = images[0]
        msks = output[0]

    # segmentation accuracy
    _, predicted = torch.max(output.data, 1)
    total_pixels += new_masks.nelement()  # number of pixels in mask
    correct_pixels += predicted.eq(new_masks.data).sum().item()  # number of pixels correctly predicted

print(np.average(losses))
print(f'Segmentation accuracy {100 * correct_pixels / total_pixels}')

msks = msks.detach().cpu()

mask = np.argmax(msks, 0)
image = imgs.detach().cpu()
print(image.shape)

DataUtils.visualise_masks(image, mask)






