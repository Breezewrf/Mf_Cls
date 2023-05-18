# -*- coding: utf-8 -*-
# @Time    : 29/4/2023 12:46 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
import argparse


def get_train_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str,
                        default=None,
                        help='Load model from a .pth file')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Downscaling factor of the images')
    parser.add_argument('--momentum', type=float, default=0.999, help='Downscaling factor of the images')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--model', type=str, default='msf', help='choose model from: unet, unetpp, msfunet, mfcls')
    parser.add_argument('--branch', type=int, default=3, help='denotes the number of streams')
    parser.add_argument('--seed', type=int, default=12321)
    parser.add_argument('--aug', type=int, default=1, help='set aug equal to 1 to implement augmentation')
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--task', type=str, default='seg')
    parser.add_argument('--dataset_name', type=str, default='prostatex')
    parser.add_argument('--branch_name', type=str, default='pre_fuse')
    parser.add_argument('--loss_f', type=str, default='focal')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--desc', type=str)
    return parser.parse_args()


def get_test_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str,
                        default="/media/breeze/dev/Mf_Cls/checkpoints/msfusion/msf_AdamW_final.pth",
                        help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--model', type=str, default='msf', help='choose model from: unet, unetpp, msfunet, mfcls')
    parser.add_argument('--branch', type=int, default=3, help='denotes the number of streams')
    parser.add_argument('--seed', type=int, default=12321)
    parser.add_argument('--aug', type=int, default=1, help='set aug equal to 1 to implement augmentation')
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--task', type=str, default='seg')
    parser.add_argument('--desc', type=str)
    return parser.parse_args()
