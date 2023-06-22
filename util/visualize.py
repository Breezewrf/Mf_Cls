# -*- coding: utf-8 -*-
# @Time    : 19/3/2023 10:22 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
import glob
import PIL.ImageDraw
import PIL.Image
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from msf_cls.backbone.pretrained import Resnet_18
from util.utils import imshow
from util.data_loading import Cls_ProstateX_Dataset
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, \
    EigenCAM, EigenGradCAM, LayerCAM, FullGrad, DeepFeatureFactorization
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib import pyplot as plt
from torch.nn import functional as F
from msf_cls.ResMSF import ResMSFNet
from util.data_loader import MSFClassifyDataset
from matplotlib import colors
from torch.nn.functional import sigmoid

cam_list = ['GradCAM', 'HiResCAM', 'GradCAMElementWise', 'GradCAMPlusPlus', 'XGradCAM', 'AblationCAM', 'ScoreCAM',
            'EigenCAM', 'EigenGradCAM', 'LayerCAM', 'FullGrad', 'mask']


def plot_imgs(images):
    # Create a 3x4 grid of subplots
    fig, axs = plt.subplots(3, 4, figsize=(10, 8))
    axs = axs.ravel()

    # Loop through each image and plot it on a subplot
    for i in range(12):
        axs[i].imshow(np.squeeze(images[i]), cmap='viridis')
        axs[i].set_title(f"{cam_list[i]}")
    plt.show()


def Generate_CAMs(model, im):
    cam = GradCAM(model=model.model, target_layers=[model.model.layer4[-1]])
    targets = [ClassifierOutputTarget(3)]
    grayscale_cam = cam(input_tensor=im.unsqueeze(0), targets=targets)
    # imshow(grayscale_cam.transpose(1, 2, 0))
    color_map = (cv2.applyColorMap((grayscale_cam * 255).astype(np.uint8).transpose(1, 2, 0), cv2.COLORMAP_JET))
    # imshow(color_map)
    map = ((im.numpy() * 255).transpose(1, 2, 0) * 0.5 + color_map * 0.5)
    # imshow(map)
    cams = [GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM,
            EigenCAM, EigenGradCAM, LayerCAM, FullGrad]
    im_np = im.numpy().transpose(1, 2, 0)
    h, w, _ = im_np.shape
    data = []
    for idx, method in enumerate(cams):
        print(cam)
        cam = method(model=model.model, target_layers=[model.model.layer4[-1]])
        grayscale_cam = cam(input_tensor=im.unsqueeze(0), targets=targets)
        # imshow(grayscale_cam.transpose(1, 2, 0), desc=str(cam))
        data.append(grayscale_cam.transpose(1, 2, 0))
    # cam_color = cv2.applyColorMap(cv2.resize(grayscale_cam.astype(np.uint8).transpose(1, 2, 0), (w, h)), cv2.COLORMAP_HSV)
    # imshow(cam_color)
    # cam_img = im_np * 255 * 0.5 + cam_color * 0.5
    # imshow(cam_img)
    # cam = ScoreCAM(model=model.model, target_layers=[model.model.layer4[-1]])
    # grayscale_cam = cam(input_tensor=im.unsqueeze(0), targets=targets)
    # imshow(grayscale_cam.transpose(1, 2, 0), desc=str(cam))
    return data


# def get_CAM_weights(model, input: torch.Tensor, last_conv_layer_name, fc_layer_name):
#     last_conv_layer = model._modules.get(last_conv_layer_name)
#     fc_layer = model._modules.get(fc_layer_name)[-1]
#     # Get the weights from the fully connected layer
#     fc_weights = np.squeeze(fc_layer.weight.data.cpu().numpy())
#     # Get the activations from the last convolutional layer
#     activations = {}
#
#     def hook_fn(name):
#         def hook(module, input, output):
#             activations[name] = output.detach()
#
#         return hook
#
#     last_conv_layer.register_forward_hook(hook_fn(last_conv_layer_name))
#     # Make a forward pass to get the model prediction
#     output = model(input)
#     pred_class = np.argmax(output.detach().numpy())
#     # Get the activations and compute the CAM weights
#     activations = activations[last_conv_layer_name].squeeze().cpu().numpy()
#     CAM_weights = fc_weights[pred_class].dot(activations.reshape(activations.shape[0], -1)).reshape(
#         activations.shape[1], activations.shape[2])
#     return CAM_weights
#
#
# def get_attention_map(img, CAM_weights):
#     # Resize the CAM weights to the size of the image
#     CAM_weights = cv2.resize(CAM_weights, (img.shape[1], img.shape[0]))
#     # Apply a ReLU activation to the CAM weights
#     CAM_weights = np.maximum(CAM_weights, 0)
#     # Normalize the CAM weights
#     CAM_weights = CAM_weights / CAM_weights.max()
#     # Convert the CAM weights to an RGB image
#     CAM_weights = np.uint8(CAM_weights)
#     CAM_weights = cv2.applyColorMap(CAM_weights, cv2.COLORMAP_HSV)
#     # Combine the CAM weights and the original image
#     attention_map = CAM_weights.astype(np.float64) * 0.5 + img.astype(np.float64) * 0.5
#     attention_map = attention_map / attention_map.max()
#     return attention_map


def get_img(id=100):
    # dataset = Cls_ProstateX_Dataset(label_dir='/media/breeze/dev/Mf_Cls/data/ProstateX/labeled_GT_colored/',
    #                                 test_mode=True)
    dataset = MSFClassifyDataset(label_dir="/media/breeze/dev/Mf_Cls/data/ProstateX/temp/predict_mask_deep/",
                                 num_classes=2, branch_num=3)
    # show image that used for training
    for id in range(20):
        im, label = dataset[id]
        imshow(im.transpose(0, 1).transpose(1, 2), desc="used for training")
        # show lesion that cropped from mask
        img = dataset.lesions[id].image_np
        imshow(img, desc="cropped lesion")

        slice_id = dataset.lesions[id].slice_id
        patient_id = dataset.lesions[id].patient_id
        path = glob.glob(
            '/media/breeze/dev/Mf_Cls/data/ProstateX/labeled_GT_colored/ProstateX-{}-{}-*'.format(
                str(patient_id).zfill(4),
                str(slice_id)))
        # '/media/breeze/dev/Mf_Cls/data/ProstateX/label_color/ProstateX-{}-{}-*.jpg'.format(patient_id, slice_id))
        mask = PIL.Image.open(path[0])

        # show the rectangle in mask
        draw = PIL.ImageDraw.Draw(mask)
        draw.rectangle((dataset.lesions[id].bbox[0][1], dataset.lesions[id].bbox[0][0], dataset.lesions[id].bbox[0][3],
                        dataset.lesions[id].bbox[0][2]))
        imshow(PIL.Image.fromarray(np.array(mask) * 255), desc="mask")

        # show the cropped mask by rectangle
        mask_ = PIL.Image.fromarray(np.uint8((mask)) * 255).crop((dataset.lesions[id].bbox[0][1],
                                                                  dataset.lesions[id].bbox[0][0],
                                                                  dataset.lesions[id].bbox[0][3],
                                                                  dataset.lesions[id].bbox[0][2]))
        imshow(mask_, desc="rectangle mask")

        CAM_hook(model, im)
        # return im, mask_


def CAM(model, im):
    pred = model(im.unsqueeze(0))
    c = np.argmax(pred.data.numpy())
    fc = model.model.fc[-1].state_dict()['weight'][c]  # shape[512]
    if len(im.shape) == 3:
        im = im.unsqueeze(0)
    im_ = model.model.conv1(im)
    im_ = model.model.bn1(im_)
    im_ = model.model.relu(im_)
    im_ = model.model.maxpool(im_)
    im_ = model.model.layer1(im_)
    im_ = model.model.layer2(im_)
    im_ = model.model.layer3(im_)
    im_ = model.model.layer4(im_)
    assert im_.shape == torch.Size([1, 512, 7, 7])
    im_flatten = im_.view(-1, 512, 7 * 7)
    activation_map = torch.zeros((7 * 7, 512))
    for i in range(0, activation_map.shape[0]):
        m = fc * im_flatten[0, :, i]
        activation_map[i] = m
    up = F.upsample_bilinear(input=activation_map.sum(dim=1).reshape(7, 7).unsqueeze(dim=0).unsqueeze(dim=0),
                             size=(64, 64))
    imshow(up[0][0], desc="CAM_1")


def CAM_hook(model, im):
    feature_data = []

    def feature_hook(model, input, output):
        feature_data.append(output.data.numpy())

    model.model._modules.get('layer4').register_forward_hook(feature_hook)
    fc_weight = model.model._modules.get('fc')[-1].weight.data.numpy()
    pred = model(im.unsqueeze(0))
    pred_c = np.argmax(pred.data.numpy())

    def makeCAM(feature, weights, classes_id):
        print(feature.shape, weights.shape, classes_id.shape)
        bz, nc, h, w = feature.shape
        cam = weights[classes_id].dot(feature.reshape(nc, h * w))
        cam = cam.reshape(h, w)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam_gray = np.uint8(255 * cam)
        return cv2.resize(cam_gray, (224, 224))

    cam = makeCAM(feature_data[0], fc_weight, pred_c)
    imshow(cam)
    im_np = im.numpy().transpose(1, 2, 0)
    h, w, _ = im_np.shape
    cam_color = cv2.applyColorMap(cv2.resize(cam, (w, h)), cv2.COLORMAP_HSV)
    imshow(cam_color)
    cam_img = im_np * 255 * 0.5 + cam_color * 0.5
    imshow(cam_img)


def get_cam_msf():
    dataset = MSFClassifyDataset(label_dir="/media/breeze/dev/Mf_Cls/data/ProstateX/test_for_cam/",
                                 num_classes=2, branch_num=3, test_mode=True)
    # show image that used for training
    for id in range(4):
        im_t2w, im_adc, im_dwi, label = dataset[id]
        im = torch.stack([im_t2w, im_adc, im_dwi])
        im = im.unsqueeze(dim=1)
        feature_data = []

        def feature_hook(model, input, output):
            feature_data.append(output.data.numpy())

        def makeCAM(feature, weights, classes_id):
            print(feature.shape, weights.shape, classes_id.shape)
            bz, nc, h, w = feature.shape
            cam = weights[classes_id].dot(feature.reshape(nc, h * w))
            cam = cam.reshape(h, w)
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            cam_gray = np.uint8(255 * cam)
            return cv2.resize(cam_gray, (224, 224))

        i = 0
        model.backbone[i]._modules.get('layer4').register_forward_hook(feature_hook)
        fc_weight = model.backbone[i]._modules.get('fc').weight.data.numpy()
        pred = model(im)
        pred_c = np.argmax(pred.data.numpy())

        cam = makeCAM(feature_data[0], fc_weight, pred_c)

        imshow(cam)

        im_np = im[i][0].numpy().transpose(1, 2, 0)
        imshow(im_np)
        h, w, _ = im_np.shape
        cam_color = cv2.applyColorMap(cv2.resize(cam, (w, h)), cv2.COLORMAP_JET)
        imshow(cam_color)
        cam_img = np.clip(im_np * 255 * 0.2 + cam_color * 0.8, 0, 255)
        imshow(cam_img)


if __name__ == '__main__':
    # Load the model
    # model = Resnet_18(num_classes=2)
    model = ResMSFNet(in_c=3, out_c=2, num_branch=3)
    # checkpoint_path = '/media/breeze/dev/Mf_Cls/checkpoints/classification/epochs[100]-bs[4]-lr[3e-05]-c2-ds[prostatex]-modal[adc]-focal/best_epoch50.pth'
    checkpoint_path = '/media/breeze/dev/Mf_Cls/checkpoints/classification/stream3-epochs[200]-bs[8]-lr[3e-05]-c2-ds[prostatex]-modal[pre_fuse]-focal-fuse4-exp1000/checkpoint_epoch200.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    # get_img()
    get_cam_msf()
    # Get the CAM weights

    # Load and preprocess the input image

    # img_tensor = preprocess(img).unsqueeze(0)
    # for i in range(20):
    #     im, mask_ = get_img(id=i)
    # img_tensor = im.unsqueeze(0)
    # CAM_weights = get_CAM_weights(model.model, img_tensor, 'layer4', 'fc')
    #
    # # Get the attention map
    # attention_map = get_attention_map(im.transpose(0, 1).transpose(1, 2).numpy(), CAM_weights)
    # imshow(attention_map)
    # CAM(model, im)
    #     CAM_hook(model, im)
    # datas = Generate_CAMs(model, im)
    # datas.append(mask_.resize((224, 224), resample=2))
    # plot_imgs(datas)
