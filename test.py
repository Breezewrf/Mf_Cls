# -*- coding: utf-8 -*-
# @Time    : 9/1/2023 7:24 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
from evaluate import evaluate
import logging
import torch
from config import get_test_args as get_args
from msf_cls.msfusion import MSFusionNet
from unetpp.unetpp_model import Nested_UNet
from unet.unet_model import UNet
from torch.utils.data import DataLoader, random_split
from util.data_loader import MSFDataset
from util.data_loading import CarvanaDataset, BasicDataset
import tqdm
import os
import numpy as np
import random

dir_t2w = 'data/ProstateX/T2W_images'
dir_adc = 'data/ProstateX/ADC_images'
dir_dwi = 'data/ProstateX/DWI_images'
dir_mask = 'data/ProstateX/labeled_GT_colored'


def test():
    args = get_args()
    test_percent = 0.2
    seed = args.seed
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info(f'testing:')
    assert (args.branch == 1 and args.model != 'msf') or (args.branch in [2, 3] and args.model == 'msf')
    if args.model == 'unet':
        model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model == 'unetpp':
        model = Nested_UNet(in_ch=1, out_ch=args.classes)
    elif args.model == 'msf':
        model = MSFusionNet(input_c=args.branch, output_c=args.classes)
    model = model.to(device)

    logging.info(f'Network:\n'
                 f'\t{args.model} model\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if args.branch != 3:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    # 1. Create dataset
    dataset = None
    if args.branch == 1:
        try:
            dataset = CarvanaDataset(dir_t2w, dir_mask, args.scale)
        except (AssertionError, RuntimeError):
            dataset = BasicDataset(dir_t2w, dir_mask, args.scale)
    elif args.branch in [2, 3]:
        dataset = MSFDataset(T2W_images_dir=dir_t2w, ADC_images_dir=dir_adc, DWI_images_dir=dir_dwi, mask_dir=dir_mask,
                             scale=args.scale, aug=args.aug, ProstateX=True)
    # 2. Split into train / validation partitions
    assert dataset is not None, f'the branch number is not set correctly: {args.branch}'
    n_test = int(len(dataset) * test_percent)
    n_train_val = len(dataset) - n_test
    train_val_set, test_set = random_split(dataset, [n_train_val, n_test],
                                           generator=torch.Generator().manual_seed(seed))
    # 1. Set `os env`
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
    # 4. Set `torch`
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_args = dict(batch_size=1, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    # Evaluation round
    val_score = evaluate(model, test_loader, device, args.amp, num_branch=args.branch, deep=args.deep)
    logging.info('Test Dice score: {}'.format(val_score))


if __name__ == '__main__':
    test()
