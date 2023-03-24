# -*- coding: utf-8 -*-
# @Time    : 10/2/2023 3:37 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
from skimage.measure import label, regionprops
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 5))


def _iou(org, bbox1, bbox2, size_average=True):
    IoU = 0.0
    im1 = np.zeros_like(org)
    im2 = np.zeros_like(org)
    im1[bbox1[0]:bbox1[2], bbox1[1]:bbox1[3]] = 1
    im2[bbox2[0]:bbox2[2], bbox2[1]:bbox2[3]] = 1
    # compute the IoU of the foreground
    Iand = np.sum((im1[:, :] * im2[:, :]))
    Ior = np.sum((im1[:, :])) + np.sum((im2[:, :])) - Iand
    IoU = Iand / Ior

    return IoU


class Slice():
    def __init__(self, path: str = '/media/breeze/dev/Mf_Cls/data/labeled_GT_colored/Lesion_37_4.png'):
        self.path = path  # labeled_GT_colored images
        self.t2w_path = path.replace("labeled_GT_colored", "T2W_images").replace("png", "jpg")
        self.adc_path = path.replace("labeled_GT_colored", "ADC_images").replace("png", "jpg")
        if 'ProstateX' in path:
            self.t2w_path = '-'.join(self.t2w_path.split('.')[0].split('-')[:-1]) + '.jpg'
            self.adc_path = '-'.join(self.adc_path.split('.')[0].split('-')[:-1]) + '.jpg'
        self.patient_id = path.split("//")[-1].split("_")[1]  # todo some bugs here
        self.np_arr = np.array(Image.open(self.path))  # gt label
        self.de_normolize()
        self.np_arr_ = None
        self.t2w_np = np.array(Image.open(self.t2w_path))
        # self.adc_np = np.array(Image.open(self.adc_path))  # todo tobe implement
        self.tumour_n = len(np.unique(self.np_arr))
        self.bbox = []  #: list[bbox: list]
        self.bbox_ex = []  #: list[bbox: list]
        self.tumour_gray = []  #: list[gray: int] corresponding to tumour_imgs, each tumour has a gray value.
        self.binary_imgs = []  #: list[img: np.array] by generate_tumour_img() -- gt binary mask
        self.tumour_imgs = []  #: list[img: np.array] by generate_tumour_img() -- tumour cropped
        self.search_expansion = 1
        self.min_search_wh = 20
        self.generate_tumour_img()
        self.generate_gray()
        # self.self_filled()

    def de_normolize(self):
        if 1 in np.unique(self.np_arr) or 2 in np.unique(self.np_arr) or 3 in np.unique(self.np_arr) or 4 in np.unique(
                self.np_arr):
            self.np_arr *= 255
        self.np_arr = np.where(self.np_arr>255, 255, self.np_arr)

    def generate_gray(self):
        assert len(self.binary_imgs) != 0
        for tumour in self.binary_imgs:
            c = np.unique(tumour)
            if len(c) > 2:
                self.self_filled()
                c = np.unique(tumour)
            assert len(c) <= 2
            self.tumour_gray.append(c.max())

    def self_filled(self):
        # done todo label_37_4's IoU is too large !!
        for i, im in enumerate(self.binary_imgs):
            _f_255 = np.count_nonzero(im - 255)
            _f_170 = np.count_nonzero(im - 170)
            _f_127 = np.count_nonzero(im - 127)
            _f_85 = np.count_nonzero(im - 85)
            _d = {
                "255": _f_255,
                "170": _f_170,
                "127": _f_127,
                "85": _f_85
            }
            d = sorted(_d.items(), key=lambda x: x[1])
            main_gray = d[0][0]
            im[im != int(main_gray)] = 0
            self.binary_imgs[i] = im

    def generate_tumour_img(self):
        bbox = self.generate_tumour_bbox()
        for box in bbox:
            self.binary_imgs.append(self.np_arr[box[0]:box[2], box[1]:box[3]])
            self.tumour_imgs.append(self.t2w_np[box[0]:box[2], box[1]:box[3]])
        ax1.imshow(self.np_arr)
        ax2.imshow(self.np_arr_)
        if len(self.binary_imgs) == 1:
            ax3.imshow(self.binary_imgs[0])
        else:
            ax3.imshow(self.binary_imgs[0])
            ax4.imshow(self.binary_imgs[1])
        plt.show()

    def generate_tumour_bbox(self):
        lbl = label(self.np_arr)
        prop = regionprops(lbl)
        np_arr_ = self.np_arr.copy()
        # print("iou: ", _iou(prop[0].bbox, prop[1].bbox))
        for prop_ in prop:
            # print('Found bbox', prop_.bbox)
            bbox_ext = self.extend_bbox(prop_.bbox)
            self.bbox_ex.append(bbox_ext)
            self.bbox.append(prop_.bbox)
        if len(self.bbox) == 1:
            cv2.rectangle(np_arr_, (self.bbox_ex[0][1], self.bbox_ex[0][0]), (self.bbox_ex[0][3], self.bbox_ex[0][2]),
                          (255, 0, 0), 1)
            self.np_arr_ = np_arr_
            return self.bbox_ex
        elif _iou(np_arr_, self.bbox_ex[0], self.bbox_ex[1]) < 0.05:
            for box in self.bbox_ex:
                cv2.rectangle(np_arr_, (box[1], box[0]), (box[3], box[2]), (255, 0, 0),
                              1)
            self.np_arr_ = np_arr_
            return self.bbox_ex
        else:
            for box in self.bbox:
                cv2.rectangle(np_arr_, (box[1], box[0]), (box[3], box[2]), (255, 0, 0),
                              1)
            self.np_arr_ = np_arr_
            return self.bbox

    def extend_bbox(self, in_box):
        """
        Extend the bounding box to define the search region
        :param in_box: a set of bounding boxes
        :param min_wh: the miniumun width/height of the search region
        """
        in_box = list(in_box)
        bbox_w = in_box[2] - in_box[0] + 1
        bbox_h = in_box[3] - in_box[1] + 1
        w_ext = int(bbox_w * (self.search_expansion / 2.))
        h_ext = int(bbox_h * (self.search_expansion / 2.))

        # todo: need to check the equation later
        min_w_ext = int((self.min_search_wh - bbox_w) / (self.search_expansion * 2.))
        min_h_ext = int((self.min_search_wh - bbox_h) / (self.search_expansion * 2.))

        w_ext = np.max((min_w_ext, w_ext))
        h_ext = np.max((min_h_ext, h_ext))
        in_box[0] -= w_ext
        in_box[1] -= h_ext
        in_box[2] += w_ext
        in_box[3] += h_ext
        # in_box[i].clip_to_image()
        for i_i, i in enumerate(in_box):
            if i < 0:
                in_box[i_i] = 0
        assert np.array(in_box).min() >= 0
        return in_box
