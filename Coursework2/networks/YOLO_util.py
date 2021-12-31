# adapted from https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-4/

from os import write
import torch

def calc_iou(box1, box2):
    """
    Calculates  IoU of two bounding boxes 
    
    box1: tensor representing first bounding box 
    box2: tensor represneting second bounding box

    returns: iou, tensor
    """

    # get coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    # get intersection rectangle
    rect_x1 =  torch.max(b1_x1, b2_x1)
    rect_y1 =  torch.max(b1_y1, b2_y1)
    rect_x2 =  torch.min(b1_x2, b2_x2)
    rect_y2 =  torch.min(b1_y2, b2_y2)
    
    # get intersection and union
    intersection = torch.clamp(rect_x2 - rect_x1 + 1, min=0) * torch.clamp(rect_y2 - rect_y1 + 1, min=0)
    union_1 = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    union_2 = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = intersection / (union_1 + union_2 - intersection)
    
    return iou

def process_results(pred, thresh=0.5, nms_thresh=0.4):
    """
    Performs object confidence thresholding and non-maximum supresssion to process 
    YOLO's prediction. Assumes there are two predictions classes (cat and dog)

    pred: tensor of shape batche size x bounding boxes x bounding box attributes
    thresh: float, confidence threshold for the bounding boxes defaults to 0.5
    nms_tresh: float, the NMS IoU threshold, defaults to 0.4
    
    returns: Dx8 tensor, containing D true detections and 8 attributes:
        index of image in the batch belonging to the prediction, 4 corner coordinates,
        objectness score, score of the class with max confidence, index of that class
    """

    # first 5 values in the last dimension represent bounding box attributes
    # set values below the confidence threshold to 0
    mask = (pred[:,:,4] > thresh).float().unsqueeze(2)
    pred = pred*mask

    # transform box attributes to 'top left corner x', 'top left corner y', 
    # 'right bottom cotrner x', right bottom corner y'
    box = pred.new(pred.shape)
    box[:,:,0] = pred[:,:,0] - pred[:,:,2] / 2
    box[:,:,1] = pred[:,:,1] - pred[:,:,3] / 2
    box[:,:,2] = pred[:,:,0] - pred[:,:,2] / 2
    box[:,:,3] = pred[:,:,1] - pred[:,:,3] / 2
    pred[:,:,:4] = box[:,:,:4]
    
    # flag to indictate whether a detection has already been made for concatenating
    detection = False

    # loop through each image in the batch to perform condidence thresholding and nms
    for idx in range(pred.shape[0]):
        img_pred = pred[idx]
        max_conf, max_conf_score = torch.max(img_pred[:,5:7], 1)  # get the max confidence out of the two possible classes
        img_pred = torch.cat((img_pred[:,:5], max_conf.unsqueeze(1), max_conf_score.unsqueeze(1)), 1)

        # remove zero values from the tensor
        img_idx = torch.nonzero(torch.Tensor(img_pred[:,4]))
        img_pred = img_pred[img_idx.squeeze(), :].reshape(-1,7) 

        # loop through each unique predicted class and perform nms
        for cls in torch.unique(img_pred[:,-1]):
            # get detections
            class_mask = img_pred * (img_pred[:,-1] == cls).unsqueeze(1)
            mask_idx = torch.nonzero(class_mask[:,-2]).squeeze()
            pred_class = img_pred[mask_idx].view(-1,7)
            # sort detections
            conf_idx = torch.sort(pred_class[:,4], descending=True)[1]
            pred_class = pred_class[conf_idx]
            indices = pred_class.shape[0]

            # loop through each detection and perform nms
            for i in range (indices):

                try: 
                    iou = calc_iou(pred_class[i].unsqueeze(0), pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break
                #print('iou.shape = ', iou)
                # remove values below the treshold
                iou_mask = (iou < nms_thresh).unsqueeze(1)
                pred_class[i+1:] = pred_class[i+1:]  * iou_mask
                mask_idx = torch.nonzero(pred_class[:,4]).squeeze()
                pred_class = pred_class[mask_idx].reshape(-1, 7)

            batch_idx = torch.ones((pred_class.shape[0], 1)) * idx
            # concanenate results to previous results if available
            if not detection:
                results = torch.cat((batch_idx, pred_class), 1)
                detection = True
            else:
                temp = torch.cat((batch_idx, pred_class), 1)
                results = torch.cat((results, temp))

    if detection:
        return results
    else:
        return None



