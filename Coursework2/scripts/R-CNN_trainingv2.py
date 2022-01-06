import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader, Dataset, Subset
import sys
sys.path.insert(1, '..') # add folder above to path for easy import 
import data_pipeline.DataUtils as DataUtils
import data_pipeline.DatasetClass as DatasetClass
from networks.Half_U_net import *
from utils.engine import train_one_epoch
from utils.utils import *
from torch.optim.lr_scheduler import StepLR

def get_instance_segmentation_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model.double()

dataset = DatasetClass.PetSegmentationDataSet('train', 'bbox', 'bin')
sub = Subset(dataset, range(0, 100))
data_loader = DataLoader(sub, 2, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler =StepLR(optimizer,step_size=3,gamma=0.1)

num_epochs = 2

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()

        
torch.save(model.state_dict(), 'rcnn_v2.pt')
