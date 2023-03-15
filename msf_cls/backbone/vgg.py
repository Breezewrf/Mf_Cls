# -*- coding: utf-8 -*-
# @Time    : 17/2/2023 7:10 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
import torch
from torch import nn
import torchvision
from torchvision.models import vgg16


class Vgg_16(nn.Module):
    def __init__(self):
        super(Vgg_16, self).__init__()
        self.in_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=2)
        self.backbone = vgg16()
        self.out_layer = nn.Conv1d(1000, 4, kernel_size=1)
        self.softmax = nn.Softmax(dim=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = self.backbone(self.in_layer(x))
        res = self.out_layer(res).transpose(0, 1)
        return self.softmax(res)
