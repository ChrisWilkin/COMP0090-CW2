import h5py as h5
import os
import numpy as np
import PIL
from PIL import Image
import math

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
    try:
        pix = np.array(file.getdata()).reshape(file.size[0], file.size[1], 3)
    except:
        print(f'Failed to convert {file_name} to array!')
        pix = []
    file.close()
    return pix

def add_margin(img, top, right, bottom, left, color):
    assert isinstance(img, Image), 'Incorrect data type'

    width, height = img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(img.mode, (new_width, new_height), color)
    result.paste(img, (left, top))
    return result

def crop_image(img, height, width):
    '''
    crops input image(PIL Image) to size height x width.
    If image is smaller than these dimensions, it is padded with black (zeros)
    '''
    

    return img

x = np.array([[1,2,3], [4,5,6],[7,8,9]])
print(x)
y = crop_image(x, 1, 8)
print(y)

def load_data(folder, test_train_val=None, indices=None, crop_size=None):
    '''
    data: images / masks / bboxes / bins - specifies the type of data to be loaded
    test_train_val: if set to 'train' or 'test' or 'val, uses a default 60:20:20 data split. Default None returns all data
    indices: if None, all data returned, otherwise accepts int or array([int]) to return specific indices of data
    crop_size: array([height, width]) to crop images to. If image is smaller, it will be padded with black
    '''
    assert folder in ['images', 'masks', 'bboxes', 'bins'], 'Invalid data category. Must be in images / masks / bboxes / bins'
    
    if test_train_val is not None:
        assert indices is None, 'test_train_val and indices cannot both be specified'
        assert test_train_val in ['test', 'train', 'val'], 'Invalid argument. Must be in test / train / val'

    elif indices is not None:
        assert isinstance(indices, np.ndarray), 'Invalid indices entry. Must by numpy array of integers'
        indices = indices.tolist()
    
    if crop_size is not None:
        assert isinstance(crop_size, np.ndarray)
        assert len(crop_size) == 2

    data = []

    if folder == 'images':
        path = paths(PATH_OG, 'images') #path to images folder
        file_names = np.array(next(os.walk(path), (None, None, []))[2])  # list of file names in folder
        jpgs = [file_names[i][-3:] == 'jpg' for i in range(len(file_names))] #ignore any non .jpg files
        file_names = file_names[jpgs]
        if indices is not None:
            file_names = [file_names[i] for i in indices]   # if indices listed, apply this

        for i, name in enumerate(file_names):
            data.append(array_from_jpg(paths(path, name)))  # load numpy array in for every filename
            # array is in WxHxC format

        for img in data:
            crop_image(img, 1,1)


    elif folder == 'masks':
        path = paths(PATH_OG, 'annotations', 'trimaps')
    elif folder == 'bboxes':
        path = paths(PATH_OG, 'annotations', 'xmls')
    elif folder == 'bins':
        path = paths(PATH_OG, 'annotations')

    return data




'''
TEST CODE


ind = np.array([1])
assert isinstance(ind, np.ndarray)
x = load_data('images', indices=ind)

'''