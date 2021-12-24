import h5py as h5
import os
import numpy as np
from PIL import Image
import math
from numpy.testing._private.utils import print_assert_equal
from xml.dom import minidom
from DataUtils import paths, save_h52
import pandas as pd

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
        pix = np.zeros((256, 256, 3))
    file.close()
    return pix

def crop_mask(img, height, width):
    '''
    crops input image(PIL Image) to size height x width.
    If image is smaller than these dimensions, it is padded with black (zeros)
    '''
    if img.shape[0] < width or img.shape[1] < height:
        w_diff = width - img.shape[0]
        h_diff = height - img.shape[1]
        w_diff = (np.abs(w_diff) + w_diff) / 2 # zero if wdiff -ve , unchanged if wdiff +ve
        h_diff = (np.abs(h_diff) + h_diff) / 2    
        right = math.ceil(w_diff / 2) 
        left = math.floor(w_diff / 2)
        top = math.ceil(h_diff / 2) 
        bottom = math.floor(h_diff / 2)
        img = np.pad(img, ((left, right), (top, bottom)), constant_values = 2)

        
    w_diff = img.shape[0] - width
    h_diff = img.shape[1] - height  
    right = img.shape[0] - math.ceil(w_diff / 2) 
    left = math.floor(w_diff / 2)
    top = math.ceil(h_diff / 2) 
    bottom = img.shape[1] - math.floor(h_diff / 2)
    img = img[left:right, top:bottom]
    return img


def array_from_png(file_name, crop=None):
    '''
    converts png file to numpy array, with cropping / padding
    '''
    file = Image.open(file_name)
    try:
        pix = np.array(file.getdata()).reshape(file.size[0], file.size[1])
    except:
        pix = np.zeros((256, 256, 1))
        print(f'array_from_png failed to convert {file_name} to array!')

    if crop is not None:
        pix = crop_mask(pix, crop[0], crop[1])
    
    pix = pix[:, :, None]
    file.close()

    return pix

def coords_from_xml(file, crop):
    '''
    extracts bounding box coordinates from xml file and accounts for image cropping
    '''
    f = minidom.parse(file, )
    xmin = int(f.getElementsByTagName('xmin')[0].firstChild.data)
    xmax = int(f.getElementsByTagName('xmax')[0].firstChild.data)
    ymin = int(f.getElementsByTagName('ymin')[0].firstChild.data)
    ymax = int(f.getElementsByTagName('ymax')[0].firstChild.data)
    width = int(f.getElementsByTagName('width')[0].firstChild.data)
    height = int(f.getElementsByTagName('height')[0].firstChild.data)

    if crop is not None:
        #print(height, crop[0])
        h_diff = height - crop[0]
        w_diff = width - crop[1]
        ymin = 0 if ymin < h_diff / 2 else ymin - np.floor(h_diff/2)
        ymax = crop[0] if ymax - crop[0] > h_diff / 2 else ymax - np.floor(h_diff/2)
        xmin = 0 if xmin < w_diff / 2 else xmin - np.floor(w_diff/2)
        xmax = crop[1] if xmax - crop[1] > w_diff / 2 else xmax - np.floor(w_diff/2)

    return (xmin, ymin, xmax, ymax)

def get_files(path, extension=None, ind=None, names=None):
    '''
    Give a path to a folder, a file extension to check for, and an optional list of indices, 
    this returns the name of all files in the folder.
    '''
    ttv = {'train': 0.6, 'test': 0.2, 'val': 0.2}

    file_names = np.array(next(os.walk(path), (None, None, []))[2])  # list of file names in folder
    if extension is not None:
        exts = [file_names[i][-3:] == extension for i in range(len(file_names))] #ignore any non .jpg files
        file_names = file_names[exts]
    if ind is not None:
        assert len(ind) <= len(file_names), f'Files: {len(file_names)}, Ind: {len(ind)}'
        file_names = [f'{names[i]}.{extension}' for i in ind]   # if indices listed, apply this

    return file_names

def get_files_bbox(path, extension=None, ind=None, dict=None):
    '''
    Give a path to a folder, a file extension to check for, and an optional list of indices, 
    this returns the name of all files in the folder.
    '''
    ttv = {'train': 0.6, 'test': 0.2, 'val': 0.2}

    file_names = np.array(next(os.walk(path), (None, None, []))[2])  # list of file names in folder
    if extension is not None:
        exts = [file_names[i][-3:] == extension for i in range(len(file_names))] #ignore any non .jpg files
        file_names = file_names[exts]
    if ind is not None:
        fls = []
        for i in ind:
            name = f'{dict[i]}.xml'
            if name in file_names:
                fls.append(name)
        file_names = np.array(fls)


    return file_names

def load_data(folder, test_train_val=None, indices=None, crop_size=None, return_dicts=False):
    '''
    folder: images / masks / bboxes / bins - specifies the type of data to be loaded
    test_train_val: if set to 'train' or 'test' or 'val, uses a default 60:20:20 data split. Default None returns all data
    indices: if None, all data returned, otherwise accepts int or array([int]) to return specific indices of data
    crop_size: array([height, width]) to crop images to. If image is smaller, it will be padded with black
    '''
    print(f'Beginning {folder} loading')
    assert folder in ['images', 'masks', 'bboxes', 'bins'], 'Invalid data category. Must be in images / masks / bboxes / bins'
    
    path = paths(PATH_OG, 'annotations', 'list.txt')
    file = pd.read_csv(path, sep=' ', skiprows=6, names=['Image', 'ID', 'Species', 'Breed'])
    df = pd.DataFrame(file)
    names, id = df['Image'], np.arange(0, len(df['Image']), 1)
    name_id = {}
    id_names = {}
    for i in range(len(names)):
        name_id[names[i]] = id[i]
        id_names[id[i]] = names[i]

    if test_train_val is not None:
        assert indices is None, 'test_train_val and indices cannot both be specified'
        assert test_train_val in ['test', 'train', 'val'], 'Invalid argument. Must be in test / train / val'
        indices = []
        if test_train_val == 'test':
            path = paths(PATH_OG, 'annotations', 'test.txt')
            test_file = pd.read_csv(path, sep=' ', skiprows=6, names=['Image', 'ID', 'Species', 'Breed']) 
            test_df = pd.DataFrame(test_file)
            for i in range(0, len(test_file['Image']), 2):
                indices.append(name_id[test_df['Image'][i]])
        if test_train_val == 'val':
            path = paths(PATH_OG, 'annotations', 'test.txt')
            test_file = pd.read_csv(path, sep=' ', skiprows=6, names=['Image', 'ID', 'Species', 'Breed']) 
            test_df = pd.DataFrame(test_file)
            for i in range(1, len(test_file['Image']), 2):
                indices.append(name_id[test_df['Image'][i]])
        if test_train_val == 'train':
            path = paths(PATH_OG, 'annotations', 'trainval.txt')
            test_file = pd.read_csv(path, sep=' ', skiprows=6, names=['Image', 'ID', 'Species', 'Breed']) 
            test_df = pd.DataFrame(test_file)
            for i in range(len(test_file['Image'])):
                indices.append(name_id[test_df['Image'][i]])
        indices = np.array(indices)

    if indices is not None:
        assert isinstance(indices, np.ndarray), 'Invalid indices entry. Must by numpy array of integers'
        indices = indices.tolist()
    
    if crop_size is not None:
        assert isinstance(crop_size, np.ndarray)
        assert len(crop_size) == 2

    data = []
    ids = []

    if folder == 'images':
        path = paths(PATH_OG, 'images') #path to images folder
        file_names = get_files(path, 'jpg', indices, id_names)

        for i, name in enumerate(file_names):
            data.append([array_from_jpg(paths(path, name), crop_size)])  # load numpy array in for every filename
            ids.append(name_id[name[:-4]])
            if (i+1) % 250 == 0:
                print(f'Loaded {i+1} / {len(file_names)}')
            # array is in WxHxC format

    elif folder == 'masks':
        path = paths(PATH_OG, 'annotations', 'trimaps')
        file_names = get_files(path, 'png', indices, id_names)
        for i, name in enumerate(file_names):
            data.append([array_from_png(paths(path, name), crop_size)])
            ids.append(name_id[name[:-4]])
            if (i+1) % 250 == 0:
                print(f'Loaded {i+1} / {len(file_names)}')
            # array is in WxHxC format

    elif folder == 'bboxes':
        path = paths(PATH_OG, 'annotations', 'xmls')
        file_names = get_files_bbox(path, 'xml', indices, id_names)
        for i, name in enumerate(file_names):
            data.append([coords_from_xml(paths(path, name), crop_size)])
            ids.append(name_id[name[:-4]])
            if (i+1) % 250 == 0:
                print(f'Loaded {i+1} / {len(file_names)}')
            # array is in WxHxC format

    elif folder == 'bins':
        path = paths(PATH_OG, 'annotations', 'list.txt')
        file = pd.read_csv(path, sep=' ', skiprows=6, names=['Image', 'ID', 'Species', 'Breed'])
        df = pd.DataFrame(file)
        if indices is not None:
            for i in indices:
                data.append([df.iloc[i]['Species'] - 1, df.iloc[i]['Breed']])
                ids.append(name_id[df.iloc[i]['Image']])
        else:
            for i in range(len(df['ID'])):
                data.append([df.iloc[i]['Species'] - 1, df.iloc[i]['Breed'], name_id[df.iloc[i]['Image']]])
                ids.append(name_id[df.iloc[i]['Image']])
    print(f'{folder} loaded successfully')   
    print('')
    if data == []:
        data = [[0]]

    if return_dicts:
        return np.array(data), np.array(ids), name_id, id_names
    else:
        return np.array(data), np.array(ids)


#a = load_data('images', test_train_val='train', crop_size=np.array([256, 256]))


#############################################################################################
######################## Apply Preprocessing and Create h5 File #############################
#############################################################################################


images = [load_data('images', test_train_val=grp, crop_size=np.array([256, 256])) for grp in ['train', 'test', 'val']]
masks = [load_data('masks', test_train_val=grp, crop_size=np.array([256, 256])) for grp in ['train', 'test', 'val']]
bboxes = [load_data('bboxes', test_train_val=grp, crop_size=np.array([256, 256])) for grp in ['train', 'test', 'val']]
bins = [load_data('bins', test_train_val=grp, crop_size=np.array([256, 256])) for grp in ['train', 'test', 'val']]
path = paths(PATH_OG, 'CustomDataset.h5')

save_h52(images, masks, bboxes, bins, path)
