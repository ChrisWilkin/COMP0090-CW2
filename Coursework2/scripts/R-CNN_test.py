import torch
from torch.utils.data import Dataset, DataLoader
import torchvision 
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import sys
import os
sys.path.append(os.path.dirname(__file__)[:-len('/scripts')])
sys.path.insert(1, '..') 
import data_pipeline.DataUtils as DataUtils
import data_pipeline.DatasetClass as DatasetClass
from networks.Half_U_net import Half_Unet
import matplotlib.pyplot as plt
import numpy as np
import cv2

dataset = DatasetClass.PetSegmentationDataSet('test', 'bbox')
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)
del dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ', device)

k=4

backbone = Half_Unet(k).to(device)
backbone.out_channels = k*4

#backbone = torchvision.models.mobilenet_v2(pretrained=True).features
#backbone.out_channels = 1280

# anchor generator
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
# feature maps for ROI cropping and ROI sizes 
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
net = FasterRCNN(backbone, num_classes=3, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).to(device)
net = net.double()

net.load_state_dict(torch.load('rcnn_unet.pt', map_location=device))

net.eval()

losses = []

selection = np.random.randint(0, len(dataloader))

detection_threshold = 0

CLASSES = ['Cat', 'Dog']

# new testing loop with updated classes
for j, data in enumerate(dataloader):
    if j % 50 == 0:
        print('image ', j)
        ims, orig_labels, orig_boxes = data.values()
        #print('ims shape', ims.shape, 'boxes shape', boxes.shape)
        ims = list(image for image in ims)

        net.eval()
        pred = net(ims)

        pred = [{k: v.to('cpu') for k, v in t.items()} for t in pred]

        if len(pred[0]['boxes']) != 0:
            boxes = pred[0]['boxes'].data.numpy()
            scores = pred[0]['scores'].data.numpy()
            #print('boxes\n', boxes)
            #print('scores\n', scores)
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
          
            print('predicted boxes\n', boxes)
        else:
            print('something went wrong... no predictions')

        print('original boxes\n', orig_boxes)


# uncomment below to run old testing loop instead
"""
for j, data in enumerate(dataloader):
    if j == 0:
        ims, boxes = data.values()
        print(ims.shape, boxes.shape)
        for i, b in enumerate(zip(ims, boxes)):
            orig_image = b[0]
            boxes_true = b[1]
            image_name = f'Image {i}'
            image = orig_image
            image /= 255
            image = torch.unsqueeze(orig_image, 0)
            
            with torch.no_grad():
                outputs = net(image.to(device)) # losses 
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            orig_image = np.transpose(np.squeeze(image.detach().cpu().numpy()), (1,2,0))

            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            
            # draw the bounding boxes and write the class name on top of it
            for k, box in enumerate(draw_boxes):
                print('prediction: ', box, pred_classes[k], scores[k], i)
                print('ground truth:', boxes_true)
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)
                cv2.putText(orig_image, pred_classes[j], 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                            2, lineType=cv2.LINE_AA)
            #cv2.imshow('Prediction', orig_image)
            #cv2.waitKey(10)
            #cv2.imwrite(f"../test_predictions/{image_name}.jpg", orig_image,)
        print(f"Image {i+1} done...")
        print('-'*50)
"""

