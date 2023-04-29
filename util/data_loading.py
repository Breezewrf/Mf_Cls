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


class Lesion():
    def __init__(self, img: np.array, p_id: int, s_id: int, grade: int, bbox: [int]):
        self.image_np = img
        self.patient_id = p_id
        self.slice_id = s_id
        self.grade = grade
        self.bbox = bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "patient_id={}, ".format(self.patient_id)
        s += "slice_id={}, ".format(self.slice_id)
        s += "grade={}), ".format(self.grade)
        s += "bbox={}".format(self.bbox)
        return s


def load_cls_prostatex_label(path='data/ProstateX/labeled_GT_colored/', num_classes=2, modal='t2w'):
    Lesion_list = []
    slices = glob(path + "*.png")
    grades = []
    assert len(slices) != 0, 'error path:{}'.format(path)
    for _, s in enumerate(slices):
        sli = Slice(s, modal=modal)
        patient_id = int(s.split('.')[0].split('-')[1])
        slice_id = int(s.split('.')[0].split('-')[2])
        grade = int(s.split('.')[0].split('-')[-1]) - 1  # for ProstateX, the grade is [1-5]
        for t_id, tumour in enumerate(sli.binary_imgs):
            assert num_classes in (2, 4, 5)
            if num_classes == 2:
                if grade in (0, 1):
                    grade = 0
                if grade in (2, 3, 4):
                    grade = 1
                le = Lesion(sli.tumour_imgs[t_id], patient_id, slice_id, grade, sli.bbox_ex)
            elif num_classes == 4:
                le = Lesion(sli.tumour_imgs[t_id], patient_id, slice_id, max(0, grade - 1),
                            sli.bbox_ex)  # max(0, grade-1)
            else:
                le = Lesion(sli.tumour_imgs[t_id], patient_id, slice_id, grade,
                            sli.bbox_ex)  # max(0, grade-1)
            Lesion_list.append(le)
            print("added")
    print("loaded!")
    return Lesion_list


def load_cls_label(path: str = "data/cls_label/twoLesion_label.txt", num_classes=2):
    Lesion_list = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            """
            each line denotes a patient,
            slices include the MRI slices of the patient
            """
            patient_id = int(float(line.split(" ")[0]))
            print(patient_id)
            slices = glob('./data/labeled_GT_colored/Lesion_' + str(patient_id) + '_*')
            print(slices)
            """
            13
            ['/media/breeze/dev/Mf_Cls/data/labeled_GT_colored/Lesion_13_1.png', '/media/breeze/dev/Mf_Cls/data/labeled_GT_colored/Lesion_13_2.png', '/media/breeze/dev/Mf_Cls/data/labeled_GT_colored/Lesion_13_3.png', '/media/breeze/dev/Mf_Cls/data/labeled_GT_colored/Lesion_13_4.png', '/media/breeze/dev/Mf_Cls/data/labeled_GT_colored/Lesion_13_5.png', '/media/breeze/dev/Mf_Cls/data/labeled_GT_colored/Lesion_13_6.png']

            Lesion_13_1.png, where 13 denotes patient id is 13, slice id is 1
            In "13", len(list) == 6 == max(slice_id), denotes patient_13 has 6 slices
            """
            grades = []
            for i in line.split(" ")[1:]:
                grades.append(int(float(i)))
            print(grades)
            """
            patient_id, grade1, grade2
            13,         2,      3
                        (127)   (255)
            denotes No.13 patient has totally two lesions
            the grade of white one is 3, the grade of gray(127) one is 2.
            Statically, there are only two lesions at most in one slice.
            """
            for s_id, s in enumerate(slices):
                sli = Slice(s)
                for t_id, tumour in enumerate(sli.binary_imgs):
                    if len(grades) == 1:
                        grade = grades[0]
                    elif len(grades) == 2:
                        # 127, 255
                        assert sli.tumour_gray[0] != 255 / 3 and sli.tumour_gray[0] != 2 * 255 / 3
                        if sli.tumour_gray[t_id] == 127:
                            grade = grades[0]
                        if sli.tumour_gray[t_id] == 255:
                            grade = grades[1]
                    else:
                        assert len(grades) == 3
                        if sli.tumour_gray[t_id] == 255 / 3:
                            grade = grades[0]
                        if sli.tumour_gray[t_id] == 2 * 255 / 3:
                            grade = grades[1]
                        if sli.tumour_gray[t_id] == 255:
                            grade = grades[2]

                    # le = Lesion(tumour, patient_id, s_id + 1, grade)  # s_id should start from 1
                    # 0, 1 - > 0; 2, 3, 4 -> 1
                    assert num_classes in (2, 4)
                    if num_classes == 2:
                        if grade in (0, 1):
                            grade = 0
                        if grade in (2, 3, 4):
                            grade = 1
                        le = Lesion(sli.tumour_imgs[t_id], patient_id, s_id + 1, grade)
                    else:
                        le = Lesion(sli.tumour_imgs[t_id], patient_id, s_id + 1, max(0, grade - 1))  # max(0, grade-1)
                    Lesion_list.append(le)
                    print("added")
    print("loaded!")
    return Lesion_list


class Cls_ProstateX_Dataset(Dataset):
    # specified for ProstateX dataset to load the GGG label
    def __init__(self, label_dir: str = 'data/ProstateX/labeled_GT_colored/', num_classes=2, branch_name='t2w',
                 test_mode=False):
        self.lesions = []
        self.num_classes = num_classes
        self.modal = branch_name
        self.lesions += load_cls_prostatex_label(path=label_dir, num_classes=num_classes, modal=self.modal)
        self.labels = [l.grade for l in self.lesions]
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
        im = self.lesions[idx].image_np
        im = enhance_cls(self, im)
        # im = torch.as_tensor(np.asarray(im).copy()).unsqueeze(dim=0)
        im = torch.as_tensor(np.asarray(im).copy())
        return im.float().contiguous(), self.lesions[idx].grade

    def __len__(self):
        return len(self.lesions)


class Cls_Dataset(Dataset):
    def __init__(self, label_dir: str = 'data/cls_label', num_classes=2, test_mode=False):
        self.label_path = glob(label_dir + "/*.txt")
        print(self.label_path)
        assert len(self.label_path) == 3
        self.lesions = []  # list of lesions in format of Lesion() object
        self.num_classes = num_classes
        for path in self.label_path:
            self.lesions += load_cls_label(path, num_classes=num_classes)
        self.labels = [l.grade for l in self.lesions]
        seed = np.random.randint(57749867)
        torch.manual_seed(seed)
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

    def __getitem__(self, idx):
        if self.transform:
            # todo implement in future
            pass
        im = self.lesions[idx].image_np
        im = enhance_cls(self, im)
        # im = torch.as_tensor(np.asarray(im).copy()).unsqueeze(dim=0)
        im = torch.as_tensor(np.asarray(im).copy())
        return im.float().contiguous(), self.lesions[idx].grade

    def __len__(self):
        return len(self.lesions)


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


def enhance_cls(self, img):
    # augmentation
    t_resize = transforms.Resize((224, 224), interpolation=Image.LANCZOS)
    if 1 == img.shape[0]:
        rgb_image = np.stack((t_resize(Image.fromarray(img)),) * 3, axis=-1)
    elif 2 == img.shape[0]:
        img = [t_resize(Image.fromarray(c)) for c in img]
        img.append(Image.fromarray((np.array(img[0]) + np.array(img[1])) / 2))
        rgb_image = np.stack(img, axis=-1)
    elif 3 == img.shape[0]:
        assert img.shape[0] == 3, "img shape is {}".format(str(img.shape))
        img = [t_resize(Image.fromarray(c)) for c in img]
        rgb_image = np.stack(img, axis=-1)
    else:
        assert len(img.shape) == 2, "img shape is {}".format(str(img.shape))
        rgb_image = np.stack((t_resize(Image.fromarray(img)),) * 3, axis=-1)
    return self.transform(rgb_image)


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
    def __init__(self, T2W_images_dir: str, ADC_images_dir: str, DWI_images_dir: str, mask_dir: str, scale: float = 1.0, aug: int = 1,
                 ProstateX=False):
        self.t2w_dir = Path(T2W_images_dir)
        self.adc_dir = Path(ADC_images_dir)
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
            mask = np.where(mask > 1, 1, 0)
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
        GGG = -1
        if self.suffix != "":
            GGG = int(str(mask_file).split('.')[0].split('-')[-1]) - 1  # GGG is range in [0, 1, 2, 3]
        assert len(t2w_file) == 1, f'Either no image or multiple images found for the ID {name} in t2w: {t2w_file}'
        assert len(adc_file) == 1, f'Either no image or multiple images found for the ID {name} in adc: {adc_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        t2w_img = load_image(t2w_file[0])
        adc_img = load_image(adc_file[0])

        assert adc_img.size == mask.size, \
            f'adc Image and mask {name} should be the same size, but are {adc_img.size} and {mask.size}, adc_img path:{adc_file[0]}'
        assert t2w_img.size == mask.size, \
            f't2w Image and mask {name} should be the same size, but are {t2w_img.size} and {mask.size}'

        t2w_img = self.preprocess(self.mask_values, t2w_img, self.scale, is_mask=False)
        adc_img = self.preprocess(self.mask_values, adc_img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        if self.aug:
            t2w_img, adc_img, mask = enhance_util(t2w_img, adc_img, mask)
        if self.suffix != "*":
            return {
                't2w_image': torch.as_tensor(t2w_img.copy()).float().contiguous(),
                'adc_image': torch.as_tensor(adc_img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous()
            }
        else:
            return {
                't2w_image': torch.as_tensor(t2w_img.copy()).float().contiguous(),
                'adc_image': torch.as_tensor(adc_img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous(),
                'GGG': GGG
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