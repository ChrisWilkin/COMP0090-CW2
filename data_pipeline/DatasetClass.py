'''
Created 19:15 21/12/2021 by Christopher Wilkin

Dataloader file containing all methods and functions relating to loading data...
'''

import h5py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import DataUtils
from DataUtils import TEST, TRAIN, VAL

h5_save_path = r"datasets/"
num_samples = 256



class PetSegmentationDataSet(Dataset):
    def __init__(self, data, mask=True, bbox=True, bin=True):
        '''
        Data: Dictionary of Images, BBoxes, Binary, Masks
        mask: whether to include mask data in dataset
        bbox: whether to include bbox data in dateset
        bin: whether to unclude binary data in dataset
        '''
        super().__init__()
        self.mask = mask
        self.bbox = bbox
        self.bin = bin
        self.data = data
        if not self.mask:
            self.data = self.data.pop('masks', None)
        if not self.bbox:
            self.data = self.data.pop('bboxes', None)
        if not self.bin:
            self.data = self.data.pop('binary', None)
        
        for key in self.data.keys():
            self.data[key] = torch.tensor(self.data[key])


    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, ind):
        if torch.is_tensor(ind):
            ind = ind.tolist()
        
        sample = {}
        for key in self.data.keys():
            sample[key] = self.data[key][ind]

        return sample


dataset = PetSegmentationDataSet(TEST, False, False, True)
  
