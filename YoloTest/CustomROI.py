import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


class CustomROI(nn.Module):
    def __init__(self, x, y, size=8):
        super(CustomROI, self).__init__()
        self.anchors = np.array([[1, 1], [2/3, 4/3], [4/3, 2/3]])
        self.scales = np.array([0.5, 0.25, 0.125])
        self.ROI = []
        for s in self.scales:
            self.ROI.append(self.region_proposal(y, x, s))
        self.ROI = np.array(self.ROI)

    def forward(self, x):
        scores = []
        boxes = []

        for box in self.ROI:
            pass
            
        return x

    def region_proposal(self, height, width, scale):
        a = []
        b = []
        for anchor in self.anchors:
            a.append(anchor * scale * np.array([width, height]))
        a = np.array(a)
        for x in range(0, width, int(1 / scale)):
            for y in range(0, height, int(1 / scale)):
                b.append(a + np.array([x,y]))

        return np.array(b)

