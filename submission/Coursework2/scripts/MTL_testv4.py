import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import time
import sys
import os
import argparse
sys.path.append(os.path.dirname(__file__)[:-len('/scripts')])
sys.path.insert(1, '..') 
import data_pipeline.DatasetClass as DatasetClass
import networks.MTL_Components as MTL
import networks.MTL_Componentsv2 as MTL2
import networks.MTL_Componentsv3 as MTL3
import data_pipeline.DataUtils as Utils

parser = argparse.ArgumentParser()
parser.add_argument('--roi', help="includes ROI in MTL, should be same as training", dest='inc_roi', action='store_true')
parser.add_argument('--no-roi', help="exludes ROI from MTL, should be same as training", dest='inc_roi', action='store_false')
parser.set_defaults(inc_roi=True)
parser.add_argument("--branch", help="branching location - middle or end", default="middle")
parser.add_argument("--thresh", help="classification threshold", type=float, default=0.5)
parser.add_argument("--body", help="filename for loading state dict of MTL body after training", type=str, default="MTL_Body.pt")
parser.add_argument("--seg", help="filename for loading state dict of segmentation after training", type=str, default="MTL_Seg.pt")
parser.add_argument("--r", help="filename for loading state dict of ROI after training", type=str, default="MTL_ROI.pt")
parser.add_argument("--iou", help="csv filename for saving IOU metrics", type=str, default="IOU.csv")
parser.add_argument("--acc", help="csv filename for saving accuracy metrics", type=str, default="Accuracy.csv")
parser.add_argument("--dataset", help="dataset to use for testing - train or test", type=str, default="test")

args = parser.parse_args()

inc_roi = args.inc_roi
if args.branch == 'middle' or args.branch == 'end':
    branch = args.branch
else:
    raise ValueError('incorrect branching specified')

if args.dataset == 'train' or args.dataset == 'test':
    data = args.dataset
else:
    raise ValueError('dataset selection invalid')

body_filename = args.body
seg_filename = args.seg
if inc_roi:
    roi_filename = args.r
else:
    print('ROI excluded...')
iou_filename = args.iou
acc_filename = args.acc

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

K = 12
CLASSES = 3 #Includes a background class = 0 for ROI
N_SEGS = 2
IN_CHANNELS = 3
THRESH = args.thresh

#Load the data in
dataset = DatasetClass.CompletePetDataSet('CompleteDataset/AllData.h5', data, 'masks', 'bins')
#sub = Subset(dataset, np.arange(1, 100, 1))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)


#Network Components
if branch == 'middle': # early branching
    body = MTL3.Body(K, IN_CHANNELS, N_SEGS).to(device).double()
    segment = MTL3.Segmentation(K, N_SEGS, body).to(device).double()
    if inc_roi:
        roi = MTL3.ROI(K, body, device)   #ROI
else: # late branching 
    body = MTL2.Body(K, IN_CHANNELS, N_SEGS).to(device).double()
    segment = MTL2.Segmentation(K, N_SEGS, body).to(device).double()
    if inc_roi:
        roi = MTL2.ROI(K, body, device)   #ROI   


#Load pretrained weights
#This looks in the Coursework2 folder by default. Change it to PATH + 'file-name.pt' to look in the networks/weights folder
PATH = os.path.dirname(__file__)[:-len('/scripts')] + '/networks/Weights/'

body.load_state_dict(torch.load(body_filename, map_location=device))
segment.load_state_dict(torch.load(seg_filename, map_location=device))
if inc_roi:
    roi.load_state_dict(torch.load(roi_filename, map_location=device))

#Set eval mode
#body.train(False)
#segment.train(False)
if inc_roi:
    roi.eval()

#Stored Data
seg_losses = []
if inc_roi:
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
        if inc_roi:
            roi_ims = list(image for image in images)

        #Forward pass -- First check that the networks work successfully
        try:
            seg_output, bin_output = segment(images)
            if inc_roi:
                roi_output = roi.forward(roi_ims)
        # except value error in case bbox values = 0
        except ValueError as e:
            print(e)
            print('skipping this batch...')
        else:
            #Retrieve predictions
            try:
                if inc_roi:
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

with open(iou_filename, 'w') as file:
    file.write('\n'.join(str(i) for i in IOU_list))
    file.write('\n'+f'{IOU_test}')

# saving accuracy at each epoch to csv file
with open(acc_filename, 'w') as file:
    file.write('\n'.join(str(i) for i in seg_accuracy ))
