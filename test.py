# -*- coding: utf-8 -*-
# @Time    : 9/1/2023 7:24 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
from evaluate import evaluate
import logging
import torch
from train import get_args
from msf_cls.msfusion import MSFusionNet
from unetpp.unetpp_model import Nested_UNet
from unet.unet_model import UNet
from torch.utils.data import DataLoader
from utils.data_loading import MSFDataset, CarvanaDataset, BasicDataset
import tqdm

t2w_test_dir = './data/test/T2W_images/'
adc_test_dir = './data/test/ADC_images/'
gt_test_dir = './data/test/labels/'


def test():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info(f'testing:')
    assert (args.branch == 1 and args.model != 'msf') or (args.branch == 2 and args.model == 'msf')
    if args.model == 'unet':
        model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model == 'unetpp':
        model = Nested_UNet(in_ch=1, out_ch=args.classes)
    elif args.model == 'msf':
        model = MSFusionNet(input_c=2, output_c=args.classes)
    model = model.to(device)

    logging.info(f'Network:\n'
                 f'\t{args.model} model\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    # 1. Create dataset
    dataset = None
    if args.branch == 1:
        try:
            dataset = CarvanaDataset(t2w_test_dir, gt_test_dir, args.scale)
        except (AssertionError, RuntimeError):
            dataset = BasicDataset(t2w_test_dir, gt_test_dir, args.scale)
    elif args.branch == 2:
        dataset = MSFDataset(t2w_test_dir, adc_test_dir, gt_test_dir, args.scale)
    loader_args = dict(batch_size=1, num_workers=2, pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=True, **loader_args)

    # Evaluation round
    val_score = evaluate(model, test_loader, device, args.amp)
    logging.info('Test Dice score: {}'.format(val_score))


if __name__ == '__main__':
    test()
