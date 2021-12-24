import h5py as h5
import os
import numpy as np
from PIL import Image
import math
from numpy.testing._private.utils import print_assert_equal
from xml.dom import minidom

PATH = f'{os.path.dirname(__file__)[:-14]}/Datasets'
HEIGHT = 256
WIDTH = 256

def paths(path, *args):
    '''
    Takes the route directory and adds folder and file name
    '''
    if len(args) == 0:
        return path 
    for a in args:
        path = f'{path}/{a}'
    return path

def load_data_from_h5(folder, file):
    '''
    opens h5 file and returns contents (shape = 256x256x3)
    (these files all only have one key)
    '''
    path = paths(PATH, folder, file)
    with h5.File(path, 'r') as file:
        key = list(file.keys())[0]
        elems = file.get(key)[:]

    return elems


### save individual arrays into h5 file as group
def save_h5(images, masks, bboxes, bins, path, group_names):
    '''
    images: list of image arrays [train_images, test_images, val_images]
    masks: list of mask arrays [train_masks, test_masks, val_masks]
    bboxes: list of bbox arrays [train_bboxes, test_bboxes, val_bboxes]
    bins: list of bin arrays [train_bins, test_bins, val_bins]
    path: file path for h5 file
    group_names: list of group_names [train, test, val]
    '''
    f = h5.File(path, "w")
    for i in range(len(group_names)):
        group = f.create_group(group_names[i])
        for j, k in enumerate(['images', 'masks', 'bboxes', 'binary']):
            subgroup = group.create_group(k)
            subgroup.create_dataset(k, data=images[i, 0].astype(np.float32), compression="gzip")
            subgroup.create_dataset("ID", data=masks[i, 1].astype(np.float32), compression="gzip")
    return
        
        
def load_group_h5(path,group_name):
    '''
    
    path: file path for h5 file
    group_name: train, test, val
    ['bboxes', 'binary', 'images', 'masks']
    '''
    with h5.File(path, 'r') as file:
        bbox = file[group_name].get('bboxes')[:]
        bin = file[group_name].get('binary')[:]
        images = file[group_name].get('images')[:]
        masks = file[group_name].get('masks')[:]

    return bbox, bin, images, masks
    
        

