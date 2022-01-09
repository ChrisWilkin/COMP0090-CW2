import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.append(os.path.dirname(__file__)[:-len('/scripts')])
sys.path.insert(1, '..') 
import data_pipeline.DatasetClass as DatasetClass
import networks.MTL_Components as MTL
import data_pipeline.DataUtils as Utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

K = 12
CLASSES = 3 #Includes a background class = 0 for ROI
N_SEGS = 2
IN_CHANNELS = 3
PATH = os.path.dirname(__file__)[:-len('/scripts')] + '/networks/Weights/'

#Load the data in
dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5', 'test', 'masks', 'bins')
sub = Subset(dataset, np.arange(1, 32, 1))
dataloader = DataLoader(sub, batch_size=8, shuffle=True, num_workers=0)


#Network Components
body = MTL.Body(K, IN_CHANNELS, N_SEGS).to(device).double()
segment = MTL.Segmentation(K, N_SEGS, body).to(device).double()
roi = MTL.ROI(K, body, device) #MTL.ROI is not actually a nn.Module class, but intiates the pytorch FasterRCNN class inside it with relevant helper functions

#Load pretrained weights
body.load_state_dict(torch.load('MTLBodyk12lr01ep1.pt', map_location=device))
segment.load_state_dict(torch.load('MTLSegk12lr01ep1.pt', map_location=device))
roi.load_state_dict(torch.load('MTLROIk12lr01ep1.pt', map_location=device))

#Set eval mode
#body.train(False)
#segment.train(False)
roi.eval()

#Stored Data
seg_losses = []
roi_losses = []
total_losses = []
seg_accuracy = []
cls_accuracy = []

###################### Testing Loop #########################

total_pixels = 0
correct_pixels = 0
total_cls = 0
correct_cls = 0

with torch.no_grad():   
    for i, data in enumerate(dataloader):
        print(f'Batch {i} / {len(dataloader)}')
        images, images_ID, masks, masks_ID, bins, bins_ID = data.values()
        images = images.to(device) / 255
        masks = masks.to(device)
        bins = bins.to(device)
        bins = bins[:, 0] + 1 #select only cat/dog data and covert to 1/2 labels

        #Seg specific setup
        masks = masks.long()

        #ROI specific setup
        roi_ims = list(image for image in images)

        #Forward pass -- First check that the networks work successfully
        try:
            seg_output = segment(images)
            roi_output = roi.forward(roi_ims)
        # except value error in case bbox values = 0
        except ValueError as e:
            print(e)
            print('skipping this batch...')
        else:
            #Retrieve predictions
            try:
                roi_scores = torch.stack([d['scores'][0] for d in roi_output], 0)
                roi_labels = torch.stack([d['labels'][0] for d in roi_output], 0)
                roi_boxes = torch.stack([d['boxes'][0] for d in roi_output], 0)
            except:
                print('No Predictions')
                sys.exit()

            # segmentation accuracy
            predicted = torch.argmax(seg_output, 1)
            print(predicted.shape)
            total_pixels += masks.nelement()  # number of pixels in mask
            correct_pixels += predicted.eq(masks.data).sum().item()

            #classification accuracy 
            pred = roi_labels
            total_cls += len(bins)
            correct_cls += torch.sum(pred == bins).item()


    # print segmentation accuracy
    seg_test_accuracy = (correct_pixels / total_pixels) * 100
    seg_accuracy.append(seg_test_accuracy)
    cls_test_accuracy = correct_cls / total_cls * 100
    cls_accuracy.append(cls_test_accuracy)

    print(seg_test_accuracy, cls_test_accuracy)

image = images[0].detach().cpu()
mask = torch.argmax(seg_output[0], 0).detach().cpu()

print(bins)
print(roi_labels)

Utils.visualise_masks(image, mask)


# saving the loss at each epoch to csv file
#with open('MTL_segment_TESTlosses.csv', 'w') as file:
#    file.write('\n'.join(str(i) for i in seg_losses))
#
#with open('MTL_ROI_TESTlosses.csv', 'w') as file:
#    file.write('\n'.join(str(i) for i in roi_losses))
#
## saving accuracy at each epoch to csv file
#with open('MTL_test_seg_accuracy.csv', 'w') as file:
#    file.write('\n'.join(str(i) for i in seg_accuracy ))

        





