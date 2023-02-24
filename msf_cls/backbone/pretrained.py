# -*- coding: utf-8 -*-
# @Time    : 23/2/2023 5:32 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
from torchvision.models import resnet18, resnet34, resnet50, \
    vgg16, ConvNeXt, VisionTransformer, SwinTransformer, \
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, \
    VGG16_Weights, ConvNeXt_Base_Weights, ViT_B_16_Weights, Swin_B_Weights
from torch import nn


class Resnet_18(nn.Module):
    def __init__(self):
        super(Resnet_18, self).__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=weights)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 4)
        self.model.add_module('softmax', nn.Softmax(dim=1))
        print("use pretrained model")

    def forward(self, x):
        return self.model(x)


class Vgg_16(nn.Module):
    def __init__(self):
        super(Vgg_16, self).__init__()
        weights = VGG16_Weights.IMAGENET1K_V1
        self.model = vgg16(weights=weights)
        for param in self.model.parameters():
            param.requires_grad = False
