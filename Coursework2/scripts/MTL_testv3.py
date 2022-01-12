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
import networks.MTL_Componentsv2 as MTL2
import networks.MTL_Componentsv3 as MTL3
import data_pipeline.DataUtils as Utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

K = 12
CLASSES = 3 #Includes a background class = 0 for ROI
N_SEGS = 2
IN_CHANNELS = 3
THRESH = 0.5

#Load the data in
dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5', 'test', 'masks', 'bins')
#sub = Subset(dataset, np.arange(1, 1000, 20))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)


#Network Components
body = MTL3.Body(K, IN_CHANNELS, N_SEGS).to(device).double()
segment = MTL3.Segmentation(K, N_SEGS, body).to(device).double()
roi = MTL3.ROI(K, body, device) #MTL.ROI is not actually a nn.Module class, but intiates the pytorch FasterRCNN class inside it with relevant helper functions

#Load pretrained weights
#This looks in the Coursework2 folder by default. Change it to PATH + 'file-name.pt' to look in the networks/weights folder
PATH = os.path.dirname(__file__)[:-len('/scripts')] + '/networks/Weights/'
body.load_state_dict(torch.load('MTL_Body_MidBranch_equalweights.pt', map_location=device))
segment.load_state_dict(torch.load('MTL_Seg_MidBranch_equalweights.pt', map_location=device))
roi.load_state_dict(torch.load('MTL_ROI_MidBranch_equalweights.pt', map_location=device))

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
IOU_testset = []
IOU_list = []
###################### Testing Loop #########################

total_pixels = 0
correct_pixels = 0
total_cls = 0
correct_cls = 0

with torch.no_grad():   
    for i, data in enumerate(dataloader):
        batch_total_pixels = 0
        batch_correct_pixels = 0
        batch_total_cls = 0
        batch_correct_cls = 0
        print(f'Batch {i+1} / {len(dataloader)}')
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
            seg_output, bin_output = segment(images)
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
            
            batch_total_pixels =  masks.nelement()  # number of pixels in mask
            total_pixels += batch_total_pixels
            
            batch_correct_pixels = predicted.eq(masks.data).sum().item()
            correct_pixels += batch_correct_pixels
            
            # segmentation IOU for each image
            batch_IOU = []
            for i in range(len(predicted)):
                intersection = torch.sum(torch.logical_and(predicted[i],masks[i]))
                union = torch.sum(torch.logical_or(predicted[i],masks[i]))
                IOU = intersection/union
                batch_IOU.append(IOU.item())
                IOU_testset.append(IOU.item())

            #classification accuracy 
            bin_output = (bin_output >= THRESH) * 1 + 1
            pred = bin_output
            batch_total_cls = len(bins)
            total_cls += batch_total_cls 
            batch_correct_cls = torch.sum(pred == bins[:, None]).item()
            correct_cls += batch_correct_cls


        # print segmentation accuracy
        seg_batch_accuracy = (batch_correct_pixels / batch_total_pixels) * 100
        seg_accuracy.append(seg_batch_accuracy)
        cls_batch_accuracy = batch_correct_cls / batch_total_cls * 100
        cls_accuracy.append(cls_batch_accuracy)
        IOU_batch = np.average(batch_IOU) * 100
        IOU_list.append(IOU_batch)
        
        

        print("Batch segmentation accuracy:",round(seg_batch_accuracy,2),"Batch classification accuracy:", cls_batch_accuracy)
        print("Average batch IOU:", round(IOU_batch,2))

        image = images[0].detach().cpu()
        mask = predicted[0].detach().cpu()

        #print(bins)
        #print(roi_labels)
        #print(bin_output)

        #Utils.visualise_MTL(image, mask, bin_output[0].detach().cpu(), roi_boxes[0].detach().cpu())
        


seg_test_accuracy = (correct_pixels / total_pixels) * 100
cls_test_accuracy = correct_cls / total_cls * 100
IOU_test = np.average(IOU_testset) * 100




print("Segmentation accuracy over entire test set:",round(seg_test_accuracy,2),"Classification accuracy over entire test set:", cls_test_accuracy)
print("Average IOU over entire test set:", round(IOU_test,2))
#saving the loss at each epoch to csv file

with open('IOU_ALL_DATA.csv', 'w') as file:
    file.write('\n'.join(str(i) for i in IOU_list))
    file.write('\n'+f'{IOU_test}')

# saving accuracy at each epoch to csv file
with open('MTL_test_seg_accuracy.csv', 'w') as file:
    file.write('\n'.join(str(i) for i in seg_accuracy ))
