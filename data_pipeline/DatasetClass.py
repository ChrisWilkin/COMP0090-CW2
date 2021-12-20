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

h5_save_path = r"datasets/"
num_samples = 256



class PetSegmentationDataSet(Dataset):
    def __init__(self, folder, *args):
        '''
        Data: Dictionary of Images, BBoxes, Binary, Masks
        mask: whether to include mask data in dataset
        bbox: whether to include bbox data in dateset
        bin: whether to unclude binary data in dataset
        '''
        super().__init__()
        self.mask = False if 'mask' not in args else True
        self.bbox = False if 'bbox' not in args else True
        self.bin = False if 'bin' not in args else True

        assert(folder in ['train', 'test', 'val'], 'Invalid folder option: must be train/test/val')
        self.folder = folder
        
        self.data = self.load_data(self.mask, self.bin, self.bbox)
        
    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, ind):
        if torch.is_tensor(ind):
            ind = ind.tolist()
        
        sample = {}
        for key in self.data.keys():
            sample[key] = self.data[key][ind]

        return sample

    def load_data(self, mask, bin, bbox):
        '''
        Selectively loads data according to what labels are specified
        '''
        img = DataUtils.load_data_from_h5(self.folder, 'images.h5')
        data = {'images':img}
        if mask:
            data['masks'] = DataUtils.load_data_from_h5(self.folder, 'masks.h5')
        if bin:
            data['bins'] = DataUtils.load_data_from_h5(self.folder, 'binary.h5')
        if bbox:
            data['bbox'] = DataUtils.load_data_from_h5(self.folder, 'bboxes.h5')

        return data


dataset = PetSegmentationDataSet('mask')

