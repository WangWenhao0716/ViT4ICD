from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import random
from collections import OrderedDict

from .gem import GeneralizedMeanPoolingP
from .metric import build_metric
from .uni_former import *

__all__ = ['Uniformer', 'uni_base']

class Uniformer(nn.Module):
    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 dev = None):
        super(Uniformer, self).__init__()
        self.pretrained = True
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.num_classes = num_classes
        
        uni = uniformer_base(pretrained=False)
        print("Loading the supervised uniformer pre-trained model")
        ckpt = torch.load('logs/pretrained/uniformer_base_in1k.pth', map_location='cpu')
        uni.load_state_dict(ckpt['model'])
        uni.head = nn.Sequential()
        
        self.base = nn.Sequential(
            uni
        )#.cuda()
        
        self.linear = nn.Linear(512, 512)
        
        self.classifier = build_metric('cos', 512, self.num_classes, s=64, m=0.35).cuda()
        self.classifier_1 = build_metric('cos', 512, self.num_classes, s=64, m=0.6).cuda()
        
        self.projector_feat_bn = nn.Sequential(
                nn.Identity()
            ).cuda()

        self.projector_feat_bn_1 = nn.Sequential(
                self.linear,
                nn.Identity()
            ).cuda()
        

    def forward(self, x, y=None):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        
        bn_x = self.projector_feat_bn(x)
        prob = self.classifier(bn_x, y)
        
        bn_x_512 = self.projector_feat_bn_1(bn_x)
        prob_1 = self.classifier_1(bn_x_512, y)
        
        
        return bn_x_512, prob, prob_1


def uni_base(**kwargs):
    return Uniformer(50, **kwargs)
