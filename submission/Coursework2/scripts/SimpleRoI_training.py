import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import sys
import os
sys.path.append(os.path.dirname(__file__)[:-len('/scripts')])
import data_pipeline.DataUtils as DataUtils
import data_pipeline.DatasetClass as DatasetClass
from networks.SimpleRoI import SimpleRoI

BATCH = 16
LR = 0.01
MOM = 0.9
EPOCHS = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = DatasetClass.PetSegmentationDataSet('train', 'bbox')
dataloader = DataLoader(data, BATCH, shuffle=True)

net = SimpleRoI().to(device)
net = net.double()

criterion = optim.SGD(net.parameters(), LR, MOM)
loss = torch.nn.MSELoss()

losses = []

for epoch in range(EPOCHS):
    for i, data in enumerate(dataloader):
        ims, boxes = data.values()
        ims = ims.to(device)
        boxes = boxes.to(device)

        pred = net(ims)
        l = loss(pred, boxes / 256)
        criterion.zero_grad()
        l.backward()
        criterion.step()
        losses.append(l.item())
        print(f'Batch {i}/{len(dataloader)}')
        print('Average Batch Loss: ', l.item())

        if (i % 5) == 0:
            print(pred[0] * 256)
            print(boxes[0])


