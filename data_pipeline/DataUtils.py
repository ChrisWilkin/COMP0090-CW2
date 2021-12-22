import h5py as h5
import os
import numpy as np
import PIL
from PIL import Image

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

##### New Code to process orignal data

PATH_OG = f'{os.path.dirname(__file__)[:-14]}/Datasets/CompleteDataset'

def array_from_jpg(file_name):
    '''
    Opens a .jpg file and returns a numpy array of H x W x C
    '''
    file = Image.open(file_name)
    pix = np.array(file.getdata()).reshape(file.size[0], file.size[1], 3)
    file.close()
    return pix

#p = paths(PATH_OG, 'images', 'Abyssinian_1.jpg')
#print(array_from_jpg(p))

def load_data(data, test_train_val=None, indices=None):
    '''
    data: images / masks / bboxes / bins - specifies the type of data to be loaded
    test_train_val: if set to 'train' or 'test' or 'val, uses a default 60:20:20 data split. Default None returns all data
    indices: if None, all data returned, otherwise accepts int or array([int]) to return specific indices of data
    '''
    assert data in ['images', 'masks', 'bboxes', 'bins'], 'Invalid data category. Must be in images / masks / bboxes / bins'
    
    if test_train_val != None:
        assert indices == None, 'test_train_val and indices cannot both be specified'
        assert test_train_val in ['test', 'train', 'val'], 'Invalid argument. Must be in test / train / val'

    elif indices != None:
        assert indices is np.ndarray, 'Invalid indices entry. Must by numpy array of integers'

    if data == 'images':
        path = paths(PATH_OG, 'images')
    elif data == 'masks':
        path = paths(PATH_OG, 'annotations', 'trimaps')
    elif data == 'bboxes':
        path = paths(PATH_OG, 'annotations', 'xmls')
    elif data == 'bins':
        path = paths(PATH_OG, 'annotations')

    return
