# -*- coding: utf-8 -*-
# @Time    : 21/3/2023 5:21 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from loss import FocalLoss
import os
import random
from msf_cls.ResMSF import ResMSFNet
from util.utils import imshow
from collections import OrderedDict
from util.visualize import *

focalLoss = FocalLoss(alpha=1, gamma=2)
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, \
    EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from msf_cls.backbone.pretrained import Resnet_18
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

cnt = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
branch_list = {'t2w': 0, 'adc': 1, 'dwi': 2}
branch = 'dwi'


class CAMs:
    def __init__(self, model, cam_name):
        self.model = model
        self.cam_name = cam_name
        self.omega = 100
        self.sigma = 0.25
        self.feature_data = []
        self.model.backbone[branch_list[branch]]._modules.get('layer4').register_forward_hook(self.feature_hook)

    def feature_hook(self, model, input, output):
        self.feature_data.append(output.data.cpu().numpy())

    def makeCAM(self, feature, weights, classes_id):
        # print(feature.shape, weights.shape, classes_id.shape)
        bz, nc, h, w = feature.shape
        cam = weights[classes_id].dot(feature.reshape(nc, h * w))
        cam = cam.reshape(h, w)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam_gray = np.uint8(255 * cam)
        return cv2.resize(cam_gray, (224, 224))

    def generate_cam_mask(self, im, im_name=''):
        self.feature_data = []

        with torch.no_grad():
            i = branch_list[branch]

            fc_weight = self.model.backbone[i]._modules.get('fc').weight.data.cpu().numpy()
            pred = self.model(im)
            pred_c = np.argmax(pred.data.cpu().numpy())

            cam = self.makeCAM(self.feature_data[0], fc_weight, pred_c)
            im_np = im[i][0].cpu().numpy().transpose(1, 2, 0)
            # imshow(im_np)
            h, w, _ = im_np.shape
            cam_color = cv2.applyColorMap(cv2.resize(cam, (w, h)), cv2.COLORMAP_JET)
            # imshow(cam_color)
            cam_img = np.clip(im_np * 255 * 0.2 + cam_color * 0.8, 0, 255)
            # imshow(cam_img)
            global cnt
            if name == '':
                Image.fromarray(cam_img.astype(np.int8), mode="RGB").save(
                    '/media/breeze/dev/Mf_Cls/data/ProstateX/cam_data/' + str(cnt) + '.jpg')
            else:
                Image.fromarray(cam_img.astype(np.int8), mode="RGB").save(
                    '/media/breeze/dev/Mf_Cls/data/ProstateX/cam_dwi_v2/' + str(im_name))
            cnt += 1
            del self.feature_data[:]
            return cam

    def generate_cam(self, im):
        logging.Logger("Warning: this function was deprecated")
        im = im[branch_list[branch]].to(device)
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        targets = [ClassifierOutputTarget(1)]
        if self.cam_name == 'FullGrad':
            cam = self.cam(model=self.model, target_layers=[])
            grayscale_cam = cam(input_tensor=im)
        else:
            cam = self.cam(model=self.model.model, target_layers=[self.model.model.layer4[-1]])
            grayscale_cam = cam(input_tensor=im, targets=targets)
        return grayscale_cam

    def generate_mask(self, im, gray_cam):
        """
        im: torch.Size([3, 224, 224])
        gray_cam: [224, 224]
        """
        if not torch.is_tensor(gray_cam):
            gray_cam = torch.tensor(gray_cam)
        if gray_cam.shape[-1] == 3:
            gray_cam = gray_cam.permute(2, 0, 1)
        if len(gray_cam.shape) != 3:
            gray_cam.unsqueeze(0)
        Ac_min = gray_cam.min()
        Ac_max = gray_cam.max()
        gray_cam = gray_cam.to(im.device)
        scaled_ac = (gray_cam - Ac_min) / (Ac_max - Ac_min)
        mask = F.sigmoid(127 * (scaled_ac - self.sigma))
        # print("mask shape:", mask.shape)
        # print("im shape:", im.shape)
        masked_img_list_t2w = []
        for idx, batch in enumerate(im[0][0]):
            masked_img_list_t2w.append(im[0][0][idx] * mask)
        masked_t2w = torch.stack(masked_img_list_t2w)
        masked_img_list_adc = []
        for idx, batch in enumerate(im[1][0]):
            masked_img_list_adc.append(im[1][0][idx] * mask)
        masked_adc = torch.stack(masked_img_list_adc)
        masked_img_list_dwi = []
        for idx, batch in enumerate(im[2][0]):
            masked_img_list_dwi.append(im[2][0][idx] * mask)
        masked_dwi = torch.stack(masked_img_list_dwi)
        masked_image = torch.stack([masked_t2w.unsqueeze(0), masked_adc.unsqueeze(0), masked_dwi.unsqueeze(0)], dim=0)
        return masked_image  # torch.tensor(3, 224, 224)


class GAIN(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet18',
                 load_dir='./checkpoints/classification/checkpoint_epoch100.pth'):
        super().__init__()
        if backbone_name == 'resnet18':
            backbone = Resnet_18(num_classes=num_classes)
        elif backbone_name == 'resmsf':
            backbone = ResMSFNet(in_c=3, out_c=2, num_branch=3)
        else:
            raise ValueError(f'Backbone "{backbone_name}" not supported')
        self.model = backbone
        state_dict = torch.load(load_dir, map_location=device)
        # 删除"model."前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'model.' in key:
                name = key.replace('model.', '')
            else:
                name = key
            new_state_dict[name] = value
        self.model.load_state_dict(new_state_dict)
        # self.gradcam = CAMs(self.model, cam_name='FullGrad')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.omega = 100
        self.sigma = 0.25
        self.feature_data = []
        self.model.backbone[branch_list[branch]]._modules.get('layer4').register_forward_hook(self.feature_hook)

    def feature_hook(self, model, input, output):
        self.feature_data.append(output.data.cpu().numpy())

    def makeCAM(self, feature, weights, classes_id):
        # print(feature.shape, weights.shape, classes_id.shape)
        bz, nc, h, w = feature.shape
        cam = weights[classes_id].dot(feature.reshape(nc, h * w))
        cam = cam.reshape(h, w)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam_gray = np.uint8(255 * cam)
        return cv2.resize(cam_gray, (224, 224))

    def generate_cam_mask(self, im):
        self.feature_data = []

        with torch.no_grad():
            i = 0
            fc_weight = self.model.backbone[i]._modules.get('fc').weight.data.cpu().numpy()
            pred = self.model(im)
            pred_c = np.argmax(pred.data.cpu().numpy())

            cam = self.makeCAM(self.feature_data[branch_list[branch]], fc_weight, pred_c)
            im_np = im[i][branch_list[branch]].cpu().numpy().transpose(1, 2, 0)
            # imshow(im_np)
            # h, w, _ = im_np.shape
            # cam_color = cv2.applyColorMap(cv2.resize(cam, (w, h)), cv2.COLORMAP_JET)
            # imshow(cam_color)
            # cam_img = np.clip(im_np * 255 * 0.2 + cam_color * 0.8, 0, 255)
            # imshow(cam_img)
            # global cnt
            # Image.fromarray(cam_img.astype(np.int8), mode="RGB").save(
            #     '/media/breeze/dev/Mf_Cls/data/ProstateX/cam_data/' + str(cnt) + '.jpg')
            # cnt += 1
            h, w, _ = im_np.shape
            cam_color = cv2.applyColorMap(cv2.resize(cam, (w, h)), cv2.COLORMAP_JET)
            cam_img = np.clip(im_np * 255 * 0.2 + cam_color * 0.8, 0, 255)

            del self.feature_data[:]
            del fc_weight
            return cam

    def generate_mask(self, im, gray_cam):
        """
        im: torch.Size([3, 224, 224])
        gray_cam: [224, 224]
        """
        if not torch.is_tensor(gray_cam):
            gray_cam = torch.tensor(gray_cam)
        if gray_cam.shape[-1] == 3:
            gray_cam = gray_cam.permute(2, 0, 1)
        if len(gray_cam.shape) != 3:
            gray_cam.unsqueeze(0)
        Ac_min = gray_cam.min()
        Ac_max = gray_cam.max()
        gray_cam = gray_cam.to(im.device)
        scaled_ac = (gray_cam - Ac_min) / (Ac_max - Ac_min)
        mask = F.sigmoid(127 * (scaled_ac - self.sigma))
        # print("mask shape:", mask.shape)
        # print("im shape:", im.shape)
        masked_img_list_t2w = []
        for idx, batch in enumerate(im[0][0]):
            masked_img_list_t2w.append(im[0][0][idx] * mask)
        masked_t2w = torch.stack(masked_img_list_t2w)
        masked_img_list_adc = []
        for idx, batch in enumerate(im[1][0]):
            masked_img_list_adc.append(im[1][0][idx] * mask)
        masked_adc = torch.stack(masked_img_list_adc)
        masked_img_list_dwi = []
        for idx, batch in enumerate(im[2][0]):
            masked_img_list_dwi.append(im[2][0][idx] * mask)
        masked_dwi = torch.stack(masked_img_list_dwi)
        masked_image = torch.stack([masked_t2w.unsqueeze(0), masked_adc.unsqueeze(0), masked_dwi.unsqueeze(0)], dim=0)
        return masked_image  # torch.tensor(3, 224, 224)

    def forward(self, x):
        im = x.to(self.device)
        # labels = labels.to(self.device)
        pred1 = self.model(im)
        # loss_pred = focalLoss(pred1, labels)

        self.model.eval()

        # code to process in batch
        masked_im_batch = torch.tensor([], device=self.device)
        with torch.no_grad():
            for batch in x.transpose(0, 1):
                batch_ = batch.unsqueeze(dim=1)
                cam = self.generate_cam_mask(batch_)
                masked_im = self.generate_mask(batch_, torch.tensor(cam))  # [3, 224, 224]
                # print(masked_im.shape)
                # masked_im = torch.ones((3, 1, 3, 224, 224), device=self.device)
                if len(masked_im_batch.shape) == 1:
                    masked_im_batch = masked_im
                else:
                    masked_im_batch = torch.cat([masked_im_batch, masked_im], dim=1)
        self.model.train()
        if len(masked_im.shape) == 3:
            masked_im = masked_im.unsqueeze(dim=0)
        else:
            masked_im = masked_im_batch
        pred2 = self.model(masked_im)
        # loss_attention = pred2[torch.arange(len(labels)), labels.to(torch.long)].sum()

        # print("labels:{}".format(labels))
        # print("pred1:{}, loss1:{},\npred2:{}, loss2:{}".format(pred1, loss_pred, pred2, loss_attention))
        # del cam
        del masked_im_batch
        del masked_im
        del batch_
        del batch
        del x
        # loss = loss_pred + loss_attention
        return pred1, pred2


if __name__ == '__main__':
    seed = 12321
    os.environ['PYTHONHASHSEED'] = str(seed)
    # (2) Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # (3) Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
    # (4) Set `torch`
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = ResMSFNet(in_c=3, out_c=2, num_branch=3).to(device)
    checkpoint_path = '/media/breeze/dev/Mf_Cls/checkpoints/classification/stream3-epochs[200]-bs[8]-lr[3e-05]-c2-ds[prostatex]-modal[pre_fuse]-focal-fuse4-exp1000-v2/checkpoint_epoch200.pth'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    dataset = MSFClassifyDataset(label_dir="/media/breeze/dev/Mf_Cls/data/ProstateX/labeled_GT_colored/",
                                 num_classes=2,
                                 branch_num=3, test_mode=True)
    cam = CAMs(model, 'ScoreCAM')
    for i in range(len(dataset)):
        im_t2w, im_adc, im_dwi, label, name = dataset[i]
        im = torch.stack([im_t2w, im_adc, im_dwi])
        im = im.unsqueeze(dim=1)

        im = im.to(device)  # torch.Size([3, 224, 224])
        # mask_ = mask_.to(device)  # PIL.Image[w, h]

        gray_cam = cam.generate_cam_mask(im, name)  # numpy.array[1, 224, 224]
        masked_im = cam.generate_mask(im, torch.tensor(gray_cam))  # torch.tensor[3, 224, 224]
        # imshow(gray_cam)
        # imshow(masked_im[branch_list[branch]][0].permute(1, 2, 0))
        pred1 = model(im)
        pred2 = model(masked_im)
        print("original image classification: {}".format(pred1))
        print("attention masked image classification: {}".format(pred2))
    # gain_model = GAINCriterionCallback(model.cuda())
    # gain_model(im, labels=torch.tensor([0]))

if __name__ == '__main__':
    model = GAIN(num_classes=2, backbone_name='resmsf',
                 load_dir='/media/breeze/dev/Mf_Cls/checkpoints/classification/stream3-epochs[200]-bs[16]-lr[3e-05]-c2-ds[prostatex]-modal[pre_fuse]-focal-fuse4-exp10/checkpoint_epoch100.pth')
    model = model.to(torch.device('cuda:0'))
    model.train()
    dataset = MSFClassifyDataset(label_dir="/media/breeze/dev/Mf_Cls/data/ProstateX/test_for_cam/", num_classes=2,
                                 branch_num=3, test_mode=True)
    im_t2w, im_adc, im_dwi, label = dataset[0]
    im = torch.stack([im_t2w, im_adc, im_dwi])
    im = im.unsqueeze(dim=1)

    im = im.to(torch.device('cuda:0'))
    model(im)
