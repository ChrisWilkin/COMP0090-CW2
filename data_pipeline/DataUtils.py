from PIL import Image
import h5py as h5
import os
import numpy as np
from numpy.lib.npyio import save
from numpy.testing._private.utils import print_assert_equal




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

def create_h5(path, name):
    h5.File(paths(path, name), 'w')
    return

def add_data_to_h5(file, name, folder, subfolder, data):
    '''
    file - h5 file to be added to: Will be created if doesnt exist
    name - name of dataset to be added
    *directory - sequence of strings indicating a group subsystem
    '''
    with file as f:
        group = f[folder]
        subgroup = group[subfolder]
        subgroup.create_dataset(name, data=data)
        print(f.keys())
    return


### save individual arrays into h5 file as group
def save_h5(ims, msks, bbs, bins, path):
    '''
    images: list of image arrays [train_images, test_images, val_images]
    masks: list of mask arrays [train_masks, test_masks, val_masks]
    bboxes: list of bbox arrays [train_bboxes, test_bboxes, val_bboxes]
    bins: list of bin arrays [train_bins, test_bins, val_bins]
    path: file path for h5 file
    group_names: list of group_names [train, test, val]
    '''
    f = h5.File(path, 'w')
    for i, grp in enumerate(['Training', 'Testing', 'Validation']):
        group = f.create_group(grp)
        Images = group.create_group('Images')
        Masks = group.create_group('Masks')
        BBoxes = group.create_group('BBoxes')
        Bins = group.create_group('Bins')

        Images.create_dataset('ims', data=ims[i][0], compression='gzip')
        Images.create_dataset('ID', data=ims[i][1], compression='gzip')
        Masks.create_dataset('masks', data=msks[i][0], compression='gzip')
        Masks.create_dataset('ID', data=msks[i][1], compression='gzip')
        BBoxes.create_dataset('bboxes', data=bbs[i][0], compression='gzip')
        BBoxes.create_dataset('ID', data=bbs[i][1], compression='gzip')
        Bins.create_dataset('bins', data=bins[i][0], compression='gzip')
        Bins.create_dataset('ID', data=bins[i][1], compression='gzip')
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
    
def load_custom_dataset(key, dataset, indices=None):
    datasets = {'Images': 'ims', 'Masks': 'masks', 'BBoxes': 'bboxes', 'Bins': 'bins'}
    assert key in ['Testing', 'Training', 'Validation']
    assert dataset in ['Images', 'Masks', 'BBoxes', 'Bins']
    path = paths(PATH, 'CompleteDataset', 'CustomDataset.h5')
    with h5.File(path, 'r') as file:
        grp = file[key]
        subgrp = grp[dataset]
        if indices is not None:
            labels = subgrp.get(datasets[dataset])[indices].astype(np.uint8)
            ids = subgrp.get('ID')[:]
    return labels, ids

def summarise_h5_structure(path):
    f = h5.File(path, 'r')

    def loop(group):
        for key in group.keys():
            if isinstance(group[key], h5.Dataset):
                print(group[key].name)
                print(group[key].shape)
            else:
                loop(group[key])

    loop(f)
    return

summarise_h5_structure(paths(PATH, 'CompleteDataset', 'AllData.h5'))