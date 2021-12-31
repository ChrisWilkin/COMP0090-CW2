import torch
import numpy as np
import PIL.Image 
from torch.autograd import Variable
import networks.YOLO as network
import networks.YOLO_util as util
import sys
import os
sys.path.append(os.path.dirname(__file__)[:-len('/networks')]) #Import other folders after this line
import data_pipeline.DataUtils as data_util

net = network.YOLO()

ims, IMids = data_util.load_custom_dataset('Training', 'Images',0)
bbs, BBOXids = data_util.load_custom_dataset('Training', 'BBoxes', 0)
bins, BINids = data_util.load_custom_dataset('Training', 'Bins', 0)

print(bbs)

imgs =  ims[0][:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
imgs = imgs[np.newaxis,:,:,:]/255.0 

input = torch.from_numpy(imgs).float()
input = Variable(input)

predictions = net(input)
predictions = util.process_results(predictions)

def loss(x, tx):
    '''
    x = pred
    tx = target
    '''
    return 0.5 * (x-tx)**2

for p in predictions:
    l = loss(p[1:5], bbs[0])
    l.backward()
    


    

