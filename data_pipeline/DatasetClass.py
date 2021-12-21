'''
Created 19:15 21/12/2021 by Christopher Wilkin

Dataloader file containing all methods and functions relating to loading data...
'''

'''
TO DO:
1. Look at coverting each file's data to correct foramt (i.e. tensors in NHWC)
2. add visualisation of images and labels
3. modify def __len__() to give comprehensive description of file lengths and data shapes etc.

'''

import torch
from torch.utils.data import Dataset, DataLoader, dataloader
import DataUtils
import time

class PetSegmentationDataSet(Dataset):
    def __init__(self, folder, *args):
        '''
        folder: the folder to take data from (test/train/val)
        *args: specifies which targets to load data from (mask, bbox, bin)
        '''
        super().__init__()
        self.mask = False if 'mask' not in args else True
        self.bbox = False if 'bbox' not in args else True
        self.bin = False if 'bin' not in args else True

        assert folder in ['train', 'test', 'val'], 'Invalid folder option: must be train/test/val'
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
        print('Loading Data...')
        t = time.time()
        img = DataUtils.load_data_from_h5(self.folder, 'images.h5')
        data = {'images':img}
        if mask:
            data['masks'] = DataUtils.load_data_from_h5(self.folder, 'masks.h5')
        if bin:
            data['bins'] = DataUtils.load_data_from_h5(self.folder, 'binary.h5')
        if bbox:
            data['bbox'] = DataUtils.load_data_from_h5(self.folder, 'bboxes.h5')
        print(f'Finished Loading Data ({time.time() - t:.2f}s)')
        return data


dataset = PetSegmentationDataSet('test', 'mask')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
print(dataset.__len__())


