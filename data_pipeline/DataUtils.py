import numpy as np
import pandas as pd
import h5py as h5
import os

PATH = f'{os.path.dirname(__file__)[:-15]}/datasets'
print(PATH)
HEIGHT = 256
WIDTH = 256

def paths(folder, file):
    '''
    Takes the route directory and adds folder and file name
    '''
    p = f'{PATH}/{folder}/{file}'
    return p

def load_data_from_h5(path, inds):
    '''
    opens h5 file and returns contents (shape = 256x256x3)
    (these files all only have one key)
    '''
    with h5.File(path, 'r') as file:
        key = list(file.keys())[0]
        elems = file[key][inds]
    return elems

TEST = {'images': load_data_from_h5(paths('test', 'images.h5')), 
        'bboxes': load_data_from_h5(paths('test', 'bboxes.h5')), 
        'binary': load_data_from_h5(paths('test', 'binary.h5')),
        'masks': load_data_from_h5(paths('test', 'masks.h5'))}
        
TRAIN = {'images': load_data_from_h5(paths('train', 'images.h5')), 
        'bboxes': load_data_from_h5(paths('train', 'bboxes.h5')), 
        'binary': load_data_from_h5(paths('train', 'binary.h5')),
        'masks': load_data_from_h5(paths('train', 'masks.h5'))}
        
VAL = {'images': load_data_from_h5(paths('val', 'images.h5')), 
        'bboxes': load_data_from_h5(paths('val', 'bboxes.h5')), 
        'binary': load_data_from_h5(paths('val', 'binary.h5')),
        'masks': load_data_from_h5(paths('val', 'masks.h5'))}

