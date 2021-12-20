import numpy as np
import pandas as pd
import h5py as h5
import os

PATH = f'{os.path.dirname(__file__)[:-14]}/datasets'
print(PATH)
HEIGHT = 256
WIDTH = 256

def paths(folder, file):
    '''
    Takes the route directory and adds folder and file name
    '''
    p = f'{PATH}/{folder}/{file}'
    return p

def load_data_from_h5(folder, file):
    '''
    opens h5 file and returns contents (shape = 256x256x3)
    (these files all only have one key)
    '''
    path = paths(folder, file)
    with h5.File(path, 'r') as file:
        key = list(file.keys())[0]
        elems = file.get(key)[:]

    return elems



