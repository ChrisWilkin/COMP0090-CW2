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

h5_save_path = r"datasets/"
num_samples = 256



class PetSegmentationDataSet(Dataset):
    def __init__(self, path, image=True, mask=True, bbox=True, bin=True):
        super().__init__()

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index: int):
        return super().__getitem__(index)
    
