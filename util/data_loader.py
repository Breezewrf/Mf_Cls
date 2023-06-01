# -*- coding: utf-8 -*-
# @Time    : 29/4/2023 12:03 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
import os
from glob import glob
from util.rumour_crop import Slice
from util.utils import imshow
from util.data_loading import unique_mask_values, load_image
from util.data_loading import load_cls_prostatex_label, enhance_cls


def enhance_util(img1, img2, img3, gt):
    # augmentation
    seed = np.random.randint(57749867)
    trans = transforms.Compose([
        transforms.ToPILImage(mode='F'),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(1),
        # transforms.RandomAutocontrast(),
        # transforms.RandomEqualize(),
        transforms.RandomRotation(15)
    ])
    torch.manual_seed(seed)
    image1 = trans(img1.astype(np.float32).transpose(1, 2, 0))

    torch.manual_seed(seed)
    image2 = trans(img2.astype(np.float32).transpose(1, 2, 0))

    torch.manual_seed(seed)
    image3 = trans(img3.astype(np.float32).transpose(1, 2, 0))

    torch.manual_seed(seed)
    gt = trans(gt.astype(np.float32))
    return np.array(image1).reshape(img1.shape), np.array(image2).reshape(img1.shape), np.array(image3).reshape(
        img1.shape), np.array(gt)


class MSFDataset(Dataset):
    def __init__(self, T2W_images_dir: str, ADC_images_dir: str, DWI_images_dir: str, mask_dir: str, num_classes = 2,
                 scale: float = 1.0,
                 aug: int = 1,
                 ProstateX=True,
                 PWH: bool = True,
                 seg: bool = True,
                 cls: bool = True):
        if seg:
            logging.info("segmentation is currently setting up")
        if cls:
            logging.info("classification is currently setting up")
        self.num_classes = num_classes
        self.seg = seg
        self.cls = cls
        self.t2w_dir = Path(T2W_images_dir)
        self.adc_dir = Path(ADC_images_dir)
        self.dwi_dir = Path(DWI_images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.ids = [splitext(file)[0] for file in listdir(T2W_images_dir) if
                    isfile(join(T2W_images_dir, file)) and not file.startswith('.')]
        self.aug = aug
        self.suffix = "*" if ProstateX else ""
        if not self.ids:
            raise RuntimeError(f'No input file found in {T2W_images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix='*'), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask=True, target_size=256):
        w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        newW = target_size
        newH = target_size
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            # normalize the mask value to integer like 0, 1, 2, 3
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            # only 0, 1 is needed here
            mask = np.where(mask > 0, 1, 0)
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.suffix + '.*'))
        t2w_file = list(self.t2w_dir.glob(name + '.*'))
        adc_file = list(self.adc_dir.glob(name + '.*'))
        dwi_file = list(self.dwi_dir.glob(name + '.*'))
        GGG = -1
        if self.suffix != "":
            GGG = int(str(mask_file).split('.')[0].split('-')[-1]) - 1  # GGG is range in [0, 1, 2, 3]
        assert len(t2w_file) == 1, f'Either no image or multiple images found for the ID {name} in t2w: {t2w_file}'
        assert len(adc_file) == 1, f'Either no image or multiple images found for the ID {name} in adc: {adc_file}'
        assert len(dwi_file) == 1, f'Either no image or multiple images found for the ID {name} in adc: {adc_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        t2w_img = load_image(t2w_file[0])
        adc_img = load_image(adc_file[0])
        dwi_img = load_image(dwi_file[0])
        assert adc_img.size == mask.size, \
            f'adc Image and mask {name} should be the same size, but are {adc_img.size} and {mask.size}, adc_img path:{adc_file[0]}'
        assert t2w_img.size == mask.size, \
            f't2w Image and mask {name} should be the same size, but are {t2w_img.size} and {mask.size}'
        assert dwi_img.size == mask.size, \
            f't2w Image and mask {name} should be the same size, but are {t2w_img.size} and {mask.size}'

        t2w_img = self.preprocess(self.mask_values, t2w_img, self.scale, is_mask=False)
        adc_img = self.preprocess(self.mask_values, adc_img, self.scale, is_mask=False)
        dwi_img = self.preprocess(self.mask_values, dwi_img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        if self.aug:
            t2w_img, adc_img, dwi_img, mask = enhance_util(t2w_img, adc_img, dwi_img, mask)
        if self.suffix != "*":
            return {
                't2w_image': torch.as_tensor(t2w_img.copy()).float().contiguous(),
                'adc_image': torch.as_tensor(adc_img.copy()).float().contiguous(),
                'dwi_image': torch.as_tensor(dwi_img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous()
            }
        else:
            return {
                't2w_image': torch.as_tensor(t2w_img.copy()).float().contiguous(),
                'adc_image': torch.as_tensor(adc_img.copy()).float().contiguous(),
                'dwi_image': torch.as_tensor(dwi_img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous(),
                'GGG': GGG,
                'name': [str(mask_file[0])]
            }


class MSFClassifyDataset(Dataset):
    def __init__(self, label_dir: str = 'data/ProstateX/labeled_GT_colored/', num_classes=2, branch_num=3,
                 test_mode=False):
        self.t2w_lesions = []
        self.adc_lesions = []
        self.dwi_lesions = []
        self.num_classes = num_classes
        self.num_branch = branch_num
        assert branch_num == 3
        self.t2w_lesions += load_cls_prostatex_label(path=label_dir, num_classes=num_classes, modal='t2w')
        self.adc_lesions += load_cls_prostatex_label(path=label_dir, num_classes=num_classes, modal='adc')
        self.dwi_lesions += load_cls_prostatex_label(path=label_dir, num_classes=num_classes, modal='dwi')
        self.labels = [l.grade for l in self.t2w_lesions]
        trans = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(1),
            transforms.RandomRotation(15),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        trans_test = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.transform = trans if not test_mode else trans_test
        assert len(self.labels) != 0, 'error dir:{}'.format(label_dir)

    def __getitem__(self, idx):
        if self.transform:
            # todo implement in future
            pass
        im_t2w = self.t2w_lesions[idx].image_np
        im_t2w = enhance_cls(self, im_t2w)
        # im = torch.as_tensor(np.asarray(im).copy()).unsqueeze(dim=0)
        im_t2w = torch.as_tensor(np.asarray(im_t2w).copy())

        im_adc = self.adc_lesions[idx].image_np
        im_adc = enhance_cls(self, im_adc)
        im_adc = torch.as_tensor(np.asarray(im_adc).copy())

        im_dwi = self.dwi_lesions[idx].image_np
        im_dwi = enhance_cls(self, im_dwi)
        im_dwi = torch.as_tensor(np.asarray(im_dwi).copy())

        return im_t2w.float().contiguous(), im_adc.float().contiguous(), im_dwi.float().contiguous(), self.t2w_lesions[idx].grade

    def __len__(self):
        return len(self.adc_lesions)
