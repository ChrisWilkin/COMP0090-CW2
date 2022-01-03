import torch
import numpy as np
import PIL.Image 
from torch.autograd import Variable
import networks.YOLO as network
import networks.YOLO_util as util
import sys
import os
sys.path.append(os.path.dirname(__file__)[:-len('/scripts')]) #Import other folders after this line
import data_pipeline.DataUtils as data_util

net = network.YOLO()

ims, IMids = data_util.load_custom_dataset('Training', 'Images', 0)
bbs, BBOXids = data_util.load_custom_dataset('Training', 'BBoxes', 0)
bins, BINids = data_util.load_custom_dataset('Training', 'Bins', 0)

print(bbs)

imgs =  ims[0][:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
imgs = imgs[np.newaxis,:,:,:]/255.0 

input = torch.tensor(imgs, requires_grad=True).float()

loss_coord = torch.nn.MSELoss()
loss_bin = torch.nn.BCELoss()
criterion = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

for i in range(100):
    predictions = net(input)
    predictions = util.process_results(predictions)
    coords = predictions[:,1:5]
    categ = predictions[:,-1]
    coords_true = torch.tensor(bbs[0], requires_grad=True).float().repeat(len(predictions), 1)
    bin_true = torch.tensor(bins[0]).float().repeat(len(predictions))
    criterion.zero_grad()
    l_coords = loss_coord(coords, coords_true)
    l_bins = loss_bin(categ, bin_true)
    loss = l_coords + l_bins
    loss.backward()
    criterion.step()
    print(loss)
    print(coords[0])








    

