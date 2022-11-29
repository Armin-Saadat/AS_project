#import os
import torch
from os.path import join
from random import randint
from typing import List, Dict, Union#, Optional, Callable, Iterable
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import RandomResizedCropVideo, RandomHorizontalFlipVideo
import warnings
from random import lognormvariate
from random import seed
import torch.nn as nn
import random

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class AddMetadata(nn.Module):
    def __init__(self, meta_size, hidden_dims, nc=4):
        super(AddMetadata, self).__init__()
        #add 3 fully connected layers where input size is size of image plus metadata
        self.fc1 = nn.Linear(meta_size+512, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, nc)

        self.dropout = nn.Dropout(0.5)
        #add softmax
        self.sftmx = nn.Softmax()

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        #print(self.fc1.weight)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        #print(self.fc2.weight)
        feat  = nn.functional.relu(self.fc3(x))
        feat = self.dropout(feat)
        #print(feat)
        out = self.sftmx(feat)
        return out        
        
    
class MetaNet(nn.Module):
    def __init__(self,  meta_size, hidden_dims, name="metanet", nc=4):
        super(MetaNet, self).__init__()
        #load_res = torch.load("/AS_clean/AS_thesis/logs/best_model.pth")
        self.model = r2plus1d_18(pretrained=False, num_classes=nc)
        #self.model.load_state_dict(load_res["model"])
        self.model.fc = Identity() #do not get the classifications from the fc layer
        #for param in self.model.parameters():
        #    param.requires_grad = False

        self.meta_module = AddMetadata(meta_size, hidden_dims, nc)
        
    def forward(self, x, meta):
        feat = self.model(x)
        meta_feat = torch.cat((feat, meta), dim=1)
        out = self.meta_module(meta_feat)
        return out
