import h5py as h5
import os
import numpy as np
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

def add_margin(img, top, right, bottom, left, color):
    '''
    adds black margin with thicknesses specified for each side
    '''

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
    if img.size[0] < width or img.size[1] < height:
        w_diff = width - img.size[0]
        h_diff = height - img.size[1]
        w_diff = (np.abs(w_diff) + w_diff) / 2 # zero if wdiff -ve , unchanged if wdiff +ve
        h_diff = (np.abs(h_diff) + h_diff) / 2    
        right = math.ceil(w_diff / 2) 
        left = math.floor(w_diff / 2)
        top = math.ceil(h_diff / 2) 
        bottom = math.floor(h_diff / 2)
        img = add_margin(img, top, right, bottom, left, (0,0,0))
    
    w_diff = img.size[0] - width
    h_diff = img.size[1] - height  
    right = img.size[0] - math.ceil(w_diff / 2) 
    left = math.floor(w_diff / 2)
    top = math.ceil(h_diff / 2) 
    bottom = img.size[1] - math.floor(h_diff / 2)
    img = img.crop((left, top, right, bottom))

    return img

def array_from_jpg(file_name, crop=None):
    '''
    Opens a .jpg file and returns a numpy array of H x W x C
    '''
    file = Image.open(file_name)
    if crop is not None:
        file = crop_image(file, crop[0], crop[1])

    try:
        pix = np.array(file.getdata()).reshape(file.size[0], file.size[1], 3)
    except:
        print(f'Failed to convert {file_name} to array!')
        pix = []
    file.close()
    return pix

def crop_mask(img, height, width):
    '''
    crops input image(PIL Image) to size height x width.
    If image is smaller than these dimensions, it is padded with black (zeros)
    '''
    print(img.shape, width, height)
    if img.shape[0] < width or img.shape[1] < height:
        print('Image too small')
        w_diff = width - img.shape[0]
        h_diff = height - img.shape[1]
        w_diff = (np.abs(w_diff) + w_diff) / 2 # zero if wdiff -ve , unchanged if wdiff +ve
        h_diff = (np.abs(h_diff) + h_diff) / 2    
        right = math.ceil(w_diff / 2) 
        left = math.floor(w_diff / 2)
        top = math.ceil(h_diff / 2) 
        bottom = math.floor(h_diff / 2)
        print(top, bottom, left, right)
        img = np.pad(img, ((left, right), (top, bottom)), constant_values = 2)
        print(img.shape)
        
    w_diff = img.shape[0] - width
    h_diff = img.shape[1] - height  
    right = img.shape[0] - math.ceil(w_diff / 2) 
    left = math.floor(w_diff / 2)
    top = math.ceil(h_diff / 2) 
    bottom = img.shape[1] - math.floor(h_diff / 2)
    img = img[left:right, top:bottom]
    return img


def array_from_png(file_name, crop=None):
    file = Image.open(file_name)
    try:
        pix = np.array(file.getdata()).reshape(file.size[0], file.size[1])
    except:
        print(f'array_from_png failed to convert {file_name} to array!')

    if crop is not None:
        pix = crop_mask(pix, crop[0], crop[1])
    
    pix = pix[:, :, None]

    return pix

def get_files(path, extension=None, ind=None, split=None):
    '''
    Give a path to a folder, a file extension to check for, and an optional list of indices, 
    this returns the name of all files in the folder.
    '''
    ttv = {'train': 0.6, 'test': 0.2, 'val': 0.2}

    file_names = np.array(next(os.walk(path), (None, None, []))[2])  # list of file names in folder
    if extension is not None:
        exts = [file_names[i][-3:] == extension for i in range(len(file_names))] #ignore any non .jpg files
        file_names = file_names[exts]
    if split is not None:   #Generates a 60/20/20 train test val split in the data
        assert split in ttv.keys(), 'Invalid split argument. Must be test, train or val'
        ind = np.arange(0, len(file_names), 1)
        ind = np.random.choice(ind, int(ttv[split] * len(file_names)))
    if ind is not None:
        assert len(ind) <= len(file_names)
        file_names = [file_names[i] for i in ind]   # if indices listed, apply this

    return file_names

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
        file_names = get_files(path, 'jpg', indices)

        for i, name in enumerate(file_names):
            data.append(array_from_jpg(paths(path, name), crop_size))  # load numpy array in for every filename
            # array is in WxHxC format

    elif folder == 'masks':
        path = paths(PATH_OG, 'annotations', 'trimaps')
        file_names = get_files(path, 'png', indices)
        for i, name in enumerate(file_names):
            data.append(array_from_png(paths(path, name), crop_size))

    elif folder == 'bboxes':
        path = paths(PATH_OG, 'annotations', 'xmls')
    elif folder == 'bins':
        path = paths(PATH_OG, 'annotations')

    return data



ind = np.array([1])
assert isinstance(ind, np.ndarray)
x = load_data('masks', indices=ind, crop_size=np.array([600, 400]))
print(x[0].shape)
print(x[0])
