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


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


def enhance_util(img1, img2, gt):
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
    gt = trans(gt.astype(np.float32))
    return np.array(image1).reshape(img1.shape), np.array(image2).reshape(img1.shape), np.array(gt)


class MSFDataset(Dataset):
    def __init__(self, T2W_images_dir: str, ADC_images_dir: str, mask_dir: str, scale: float = 1.0, aug: int = 1):
        self.t2w_dir = Path(T2W_images_dir)
        self.adc_dir = Path(ADC_images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.ids = [splitext(file)[0] for file in listdir(T2W_images_dir) if
                    isfile(join(T2W_images_dir, file)) and not file.startswith('.')]
        self.aug = aug
        if not self.ids:
            raise RuntimeError(f'No input file found in {T2W_images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=''), self.ids),
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
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

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
        mask_file = list(self.mask_dir.glob(name + '.*'))
        t2w_file = list(self.t2w_dir.glob(name + '.*'))
        adc_file = list(self.adc_dir.glob(name + '.*'))

        assert len(t2w_file) == 1, f'Either no image or multiple images found for the ID {name} in t2w: {t2w_file}'
        assert len(adc_file) == 1, f'Either no image or multiple images found for the ID {name} in adc: {adc_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        t2w_img = load_image(t2w_file[0])
        adc_img = load_image(adc_file[0])

        assert adc_img.size == mask.size, \
            f'adc Image and mask {name} should be the same size, but are {adc_img.size} and {mask.size}'
        assert t2w_img.size == mask.size, \
            f't2w Image and mask {name} should be the same size, but are {t2w_img.size} and {mask.size}'

        t2w_img = self.preprocess(self.mask_values, t2w_img, self.scale, is_mask=False)
        adc_img = self.preprocess(self.mask_values, adc_img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        if self.aug:
            t2w_img, adc_img, mask = enhance_util(t2w_img, adc_img, mask)
        return {
            't2w_image': torch.as_tensor(t2w_img.copy()).float().contiguous(),
            'adc_image': torch.as_tensor(adc_img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

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
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix="")
