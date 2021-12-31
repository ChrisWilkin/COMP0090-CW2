import torch
import numpy as np
import networks.YOLO as network
import networks.YOLO_util as util
import sys
import os
sys.path.append(os.path.dirname(__file__)[:-len('/networks')]) #Import other folders after this line
import data_pipeline.DataUtils as data_util

net = network.YOLO()

ims, IMids = data_util.load_custom_dataset('Training', 'Images')
bbs, BBOXids = data_util.load_custom_dataset('Training', 'BBoxes')
bins, BINids = data_util.load_custom_dataset('Training', 'Bins')

print(IMids[:5])
print(BBOXids[:5])
print(BINids[:5])

