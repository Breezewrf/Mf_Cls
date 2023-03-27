# -*- coding: utf-8 -*-
# @Time    : 21/3/2023 5:21 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from loss import FocalLoss

focalLoss = FocalLoss(alpha=1, gamma=2)
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, \
    EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from msf_cls.backbone.pretrained import Resnet_18
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cams = [GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM,
        EigenCAM, EigenGradCAM, LayerCAM, FullGrad]


class CAMs:
    def __init__(self, model, cam_name):
        self.cam_list = {'GradCAM': GradCAM, 'HiResCAM': HiResCAM, 'GradCAMElementWise': GradCAMElementWise,
                         'GradCAMPlusPlus': GradCAMPlusPlus, 'XGradCAM': XGradCAM, 'AblationCAM': AblationCAM,
                         'ScoreCAM': ScoreCAM, 'EigenCAM': EigenCAM, 'EigenGradCAM': EigenGradCAM, 'LayerCAM': LayerCAM,
                         'FullGrad': FullGrad}
        self.model = model
        self.cam_name = cam_name
        assert cam_name in self.cam_list, "{} is not implemented yet".format(cam_name)
        self.cam = self.cam_list[cam_name]
        self.omega = 100
        self.sigma = 0.25

    def generate_cam(self, im):
        im = im.to(device)
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        targets = [ClassifierOutputTarget(3)]
        if self.cam_name == 'FullGrad':
            cam = self.cam(model=self.model.model,target_layers=[])
            grayscale_cam = cam(input_tensor=im)
        else:
            cam = self.cam(model=self.model.model, target_layers=[self.model.model.layer4[-1]])
            grayscale_cam = cam(input_tensor=im, targets=targets)
        return grayscale_cam

    def generate_mask(self, im, gray_cam):
        """
        im: torch.Size([3, 224, 224])
        gray_cam: [1, 224, 224]
        """
        if not torch.is_tensor(gray_cam):
            gray_cam = torch.tensor(gray_cam)
        if gray_cam.shape[-1] == 3:
            gray_cam = gray_cam.permute(2, 0, 1)

        Ac_min = gray_cam.min()
        Ac_max = gray_cam.max()
        gray_cam = gray_cam.to(im.device)
        scaled_ac = (gray_cam - Ac_min) / (Ac_max - Ac_min)
        mask = F.sigmoid(self.omega * (scaled_ac - self.sigma))
        # print("mask shape:", mask.shape)
        # print("im shape:", im.shape)
        masked_img_list = []
        for idx, batch in enumerate(mask):
            masked_img_list.append(im[idx] - im[idx] * mask[idx].unsqueeze(dim=0))
        masked_image = torch.stack(masked_img_list)
        return masked_image  # torch.tensor(3, 224, 224)


class GAIN(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet18',
                 load_dir='./checkpoints/classification/checkpoint_epoch100.pth'):
        super().__init__()
        if backbone_name == 'resnet18':
            backbone = Resnet_18(num_classes=num_classes)
        else:
            raise ValueError(f'Backbone "{backbone_name}" not supported')
        self.model = backbone
        state_dict = torch.load(load_dir, map_location=device)
        self.model.load_state_dict(state_dict)
        self.gradcam = CAMs(self.model, cam_name='FullGrad')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, labels):
        im = x.to(self.device)
        labels = labels.to(self.device)
        pred1 = self.model(im)
        loss_pred = focalLoss(pred1, labels)

        self.model.eval()
        cam = self.gradcam.generate_cam(x)
        masked_im = self.gradcam.generate_mask(x, torch.tensor(cam))  # [3, 224, 224]
        self.model.train()
        if len(masked_im.shape) == 3:
            masked_im = masked_im.unsqueeze(dim=0)
        pred2 = self.model(masked_im)
        loss_attention = pred2[torch.arange(len(labels)), labels.to(torch.long)].sum()
        # print("labels:{}".format(labels))
        # print("pred1:{}, loss1:{},\npred2:{}, loss2:{}".format(pred1, loss_pred, pred2, loss_attention))

        loss = loss_pred + loss_attention
        return loss


class GAINCriterionCallback:
    def __init__(self, model, num_classes=4, cam_name='FullGrad'):
        self.model = model
        self.num_classes = num_classes
        self.cam = CAMs(model, cam_name=cam_name)
        self.device = device

    def __call__(self, x, labels):
        if len(x.shape) == 3:
            im = x.unsqueeze(dim=0)
        else:
            im = x
        self.model.zero_grad()
        im = im.to(self.device)
        labels = labels.to(self.device)
        pred1 = self.model(im)
        loss_pred = focalLoss(pred1, labels)

        self.model.eval()
        cam = self.cam.generate_cam(x)
        masked_im = self.cam.generate_mask(x, torch.tensor(cam))  # [3, 224, 224]
        self.model.train()
        if len(masked_im.shape) == 3:
            masked_im = masked_im.unsqueeze(dim=0)
        pred2 = self.model(masked_im)
        loss_attention = focalLoss(pred2, labels)
        loss = loss_pred + loss_attention
        return loss


from util.visualize import *

if __name__ == '__main__':
    model = Resnet_18(num_classes=4).to(device)
    checkpoint_path = '/media/breeze/dev/Mf_Cls/checkpoints/classification/checkpoint_epoch100.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    im, mask_ = get_img(id=54)  # id=45 is an ideal example
    im = im.to(device)  # torch.Size([3, 224, 224])
    # mask_ = mask_.to(device)  # PIL.Image[w, h]
    cam = CAMs(model, 'FullGrad')
    gray_cam = cam.generate_cam(im)  # numpy.array[1, 224, 224]
    masked_im = cam.generate_mask(im, torch.tensor(gray_cam))  # torch.tensor[3, 224, 224]
    pred1 = model(im.unsqueeze(dim=0))
    pred2 = model(masked_im.unsqueeze(dim=0))
    print("original image classification: {}".format(pred1))
    print("attention masked image classification: {}".format(pred2))
    # gain_model = GAINCriterionCallback(model.cuda())
    # gain_model(im, labels=torch.tensor([0]))
