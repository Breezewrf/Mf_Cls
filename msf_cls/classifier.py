# -*- coding: utf-8 -*-
# @Time    : 10/2/2023 10:48 AM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
from backbone.resnet import *
import torch
import torch.nn as nn

module_dict = {
    "res18": resnet18(),
    "res34": resnet34(),
    "res50": resnet50(),
    "res101": resnet101(),

}


class Classifier(nn.Module):
    def __init__(self, backbone: str = "res18"):
        super(Classifier, self).__init__()
        self.backbone = resnet34()

    def forward(self, image, mask):
        pass

