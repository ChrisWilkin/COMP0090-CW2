import numpy as np
import pandas as pd
import h5py as h5
import os
'''
helper functions for loading in data for the dataset class (merged)
''' 
cur_path = os.path.dirname(__file__)
print(cur_path)
PATH = cur_path[:-14] #this is absolutely disgusting, dont judge (removes /data_pipeline from the directory)
PATH = f'{PATH}/datasets'


def paths(folder, file):
    '''
    Takes the route director and adds folder and file name
    '''
    p = f'{PATH}/{folder}/{file}'
    return p

def load_data_from_h5(path, inds):
    with h5.File(path, 'r') as file:
        key = list(file.keys())[0]
        elems = file[key][inds]
    return elems

p = paths('test', 'images.h5')
print(p)
f = h5.File(p)
