import torch
from torch import nn
from einops import rearrange
from efficientnet_pytorch import EfficientNet
import cv2
import re
import numpy as np
from torch import einsum
from random import randint

import timm

from torchsummary import summary

class Baseline(nn.Module):
    def __init__(self, config):
        super().__init__() 
        
        self.dim = config['model']['dim']
        self.mlp_dim = config['model']['mlp-dim']
        
        self.num_classes = config['model']['num-classes']

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.mlp_head = nn.Sequential(
            nn.Linear(self.dim, self.mlp_dim),
            nn.Linear(self.mlp_dim, self.num_classes)
        )
    
        for index, (name, param) in enumerate(self.mlp_head.named_parameters()):
            param.requires_grad = True


    def forward(self, x, mask=None): # (B x C x H x W)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return self.mlp_head(x)
