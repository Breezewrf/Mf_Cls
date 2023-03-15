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
    def __init__(self, num_classes=2):
        super(Resnet_18, self).__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=weights)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)
        print("use pretrained model")
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        res = self.softmax(x)
        return res


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        weights = 'vgg16_bn'
        self.model = vgg16(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.Dropout(0.5)
        )
        self.softmax = nn.Softmax(dim=1)
        print("use pretrained model")
        # for param in self.model.features.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        res = self.softmax(x)
        return res






