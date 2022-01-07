'''
Created 19:15 21/12/2021 by Christopher Wilkin

Dataloader file containing all methods and functions relating to loading data...
'''

'''
TO DO:
1. Look at coverting each file's data to correct foramt (i.e. tensors in NHWC)
2. add visualisation of images and labels
3. modify def __len__() to give comprehensive description of file lengths and data shapes etc.
4. can we combine all the h5 files into one? Is it faster?
'''

import torch
from torch.utils.data import Dataset, DataLoader, dataloader
import data_pipeline.DataUtils as DataUtils
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

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
        img = torch.from_numpy(img).permute(0,3,1,2)
        data = {'images':img}
        
        if mask:
            mask = DataUtils.load_data_from_h5(self.folder, 'masks.h5')
            mask = torch.from_numpy(mask).permute(0,3,1,2)
            data['masks'] = mask
        if bin:
            bins = torch.from_numpy(DataUtils.load_data_from_h5(self.folder, 'binary.h5'))
            bins = bins + 1 # classes are 1,2 instead of 0,1
            data['bins'] = bins
        if bbox:
            box = torch.from_numpy(DataUtils.load_data_from_h5(self.folder, 'bboxes.h5'))
            data['bbox'] = box
        print(f'Finished Loading Data ({time.time() - t:.2f}s)')
        return data
    
    def visualize_data(self):
        index = np.random.randint(self.data['images'].size()[0]-1)
        img = self.data['images'][index].permute(1,2,0).numpy()/255
        fig, ax = plt.subplots()
        if self.mask:
            msk = self.data['masks'][index].permute(1,2,0).numpy()
            msk = np.ones(msk.shape)-msk
            img = img - msk
            
        ax.imshow(img)
        
        if self.bbox:
            box = np.round(self.data['bbox'][index])
            x = box[0]
            y = box[1]
            width = box[2] - box[0]
            height = box[3] - box[1]
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        if self.bin:
            label = "dog" if self.data['bins'][index] == 0 else "cat"
            fig.text(0.25, 0.80,label,fontsize = 10,bbox ={'facecolor':'white','alpha':0.6,'pad':10})
        
        # remove one of these
        fig.show()
        plt.show()
        
        
        
class CompletePetDataSet(Dataset):
    def __init__(self, file, group_name, *args):
        '''
        folder: the folder to take data from (test/train/val)
        *args: specifies which targets to load data from (mask, bbox, bins)
        '''
        super().__init__()
        self.file = file
        self.mask = False if 'masks' not in args else True
        self.bbox = False if 'bboxes' not in args else True
        self.bin = False if 'bins' not in args else True

        assert group_name in ['train', 'test', 'val'], 'Invalid group option: must be train/test/val'
        self.group_name = group_name
        
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
        img, img_ID = DataUtils.load_group_h5(self.file,self.group_name, 'images')
        img = torch.from_numpy(img).permute(0,3,1,2)
        data = {'images':img,
                'images_ID': img_ID}
        if mask:
            masks, masks_ID = DataUtils.load_group_h5(self.file,self.group_name,'masks')
            masks[masks==3]=1
            masks[masks==2]=0
            masks = torch.from_numpy(masks)
            data['masks'] = masks
            data['masks_ID'] = masks_ID
        if bin:
            bins, bins_ID = DataUtils.load_group_h5(self.file,self.group_name,'bins')
            bins = torch.from_numpy(bins)
            data['bins'] = bins
            data['bins_ID'] = bins_ID
            
        if bbox:
            bboxes, bboxes_ID = DataUtils.load_group_h5(self.file,self.group_name,'bboxes')
            bboxes = torch.from_numpy(bboxes)
            data['bboxes'] = bboxes
            data['bboxes_ID'] = bboxes_ID
        print(f'Finished Loading Data ({time.time() - t:.2f}s)')
        return data
    
    def visualize_data(self):
        if self.bbox:
            ID = np.random.choice(self.data['bboxes_ID'])
            image_index = np.where(self.data['images_ID'] == ID)[0][0]
            bbox_index = np.where(self.data['bboxes_ID'] == ID)[0][0]
        else:
            image_index = np.random.randint(len(self.data['images_ID'])-1)
            
        
        
        img = self.data['images'][image_index].permute(1,2,0).numpy()/255
        fig, ax = plt.subplots()
        if self.mask:
            mask_index = image_index
            msk = self.data['masks'][mask_index].numpy().reshape(256,256,1)
            msk = np.ones(msk.shape)-msk
            img = img - msk
            
        ax.imshow(img)
        
        if self.bbox:
            box = np.round(self.data['bboxes'][bbox_index])
            x = box[0]
            y = box[1]
            width = box[2] - box[0]
            height = box[3] - box[1]
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        if self.bin:
            bin_index = image_index
            label = "dog" if self.data['bins'][bin_index][0] == 1 else "cat"
            fig.text(0.25, 0.80,label,fontsize = 10,bbox ={'facecolor':'white','alpha':0.6,'pad':10})
        
        # remove one of these
        fig.show()
        plt.show()
        
        
class CompletePetDataSet(Dataset):
    def __init__(self, file, group_name, *args):
        '''
        folder: the folder to take data from (test/train/val)
        *args: specifies which targets to load data from (mask, bbox, bins)
        '''
        super().__init__()
        self.file = file
        self.mask = False if 'masks' not in args else True
        self.bbox = False if 'bboxes' not in args else True
        self.bin = False if 'bins' not in args else True

        assert group_name in ['train', 'test', 'val'], 'Invalid group option: must be train/test/val'
        self.group_name = group_name
        
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
        img, img_ID = DataUtils.load_group_h5(self.file,self.group_name, 'images')
        img = torch.from_numpy(img).permute(0,3,1,2)
        data = {'images':img,
                'images_ID': img_ID}
        if mask:
            masks, masks_ID = DataUtils.load_group_h5(self.file,self.group_name,'masks')
            masks[masks==3]=1
            masks[masks==2]=0
            masks = torch.from_numpy(masks)
            data['masks'] = masks
            data['masks_ID'] = masks_ID
        if bin:
            bins, bins_ID = DataUtils.load_group_h5(self.file,self.group_name,'bins')
            bins = torch.from_numpy(bins)
            data['bins'] = bins
            data['bins_ID'] = bins_ID
            
        if bbox:
            bboxes, bboxes_ID = DataUtils.load_group_h5(self.file,self.group_name,'bboxes')
            bboxes = torch.from_numpy(bboxes)
            data['bboxes'] = bboxes
            data['bboxes_ID'] = bboxes_ID
        print(f'Finished Loading Data ({time.time() - t:.2f}s)')
        return data
    
    def visualize_data(self):
        if self.bbox:
            ID = np.random.choice(self.data['bboxes_ID'])
            image_index = np.where(self.data['images_ID'] == ID)[0][0]
            bbox_index = np.where(self.data['bboxes_ID'] == ID)[0][0]
        else:
            image_index = np.random.randint(len(self.data['images_ID'])-1)
            
        
        
        img = self.data['images'][image_index].permute(1,2,0).numpy()/255
        fig, ax = plt.subplots()
        if self.mask:
            mask_index = image_index
            msk = self.data['masks'][mask_index].numpy().reshape(256,256,1)
            msk = np.ones(msk.shape)-msk
            img = img - msk
            
        ax.imshow(img)
        
        if self.bbox:
            box = np.round(self.data['bboxes'][bbox_index])
            x = box[0]
            y = box[1]
            width = box[2] - box[0]
            height = box[3] - box[1]
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        if self.bin:
            bin_index = image_index
            label = "dog" if self.data['bins'][bin_index][0] == 1 else "cat"
            fig.text(0.25, 0.80,label,fontsize = 10,bbox ={'facecolor':'white','alpha':0.6,'pad':10})
        
        # remove one of these
        fig.show()
        plt.show()
        


#dataset = PetSegmentationDataSet('test', 'mask')
#dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
#print(dataset.__len__())
#dataset.visualize_data()


