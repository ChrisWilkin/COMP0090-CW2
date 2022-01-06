import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for data in metric_logger.log_every(data_loader, print_freq, header):
        images, labels, boxes = data.values()
        labels = labels.to(torch.int64)
                
        images = list(image.to(device) for image in images)
        targets = [{'boxes': boxes[i].reshape((1,4)).to(device), 'labels': labels[i].reshape(1).to(device)} for i in range(len(labels))]

        try:
            print('targets = ', targets)
            print('images = ', images)
            loss_dict = model(images, targets)
  
        # except value error in case bbox values = 0
        except ValueError as e:
          print(e)
          print('skipping this batch...')

        else:
          losses = sum(loss for loss in loss_dict.values())
          # reduce losses over all GPUs for logging purposes
          loss_dict_reduced = utils.reduce_dict(loss_dict)
          losses_reduced = sum(loss for loss in loss_dict_reduced.values())

          loss_value = losses_reduced.item()

          if not math.isfinite(loss_value):
              print(f"Loss is {loss_value}, stopping training")
              print(loss_dict_reduced)
              sys.exit(1)

          optimizer.zero_grad()
          if scaler is not None:
              scaler.scale(losses).backward()
              scaler.step(optimizer)
              scaler.update()
          else:
              losses.backward()
              optimizer.step()

          metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
          metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types