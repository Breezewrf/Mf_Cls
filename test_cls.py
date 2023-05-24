# -*- coding: utf-8 -*-
# @Time    : 15/3/2023 6:55 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
import torch
import logging
from msf_cls.backbone.pretrained import Resnet_18, VGG16
from tqdm import tqdm
from util.data_loading import Cls_Dataset
from torch.utils.data import DataLoader, random_split
import argparse
import numpy as np
from util.utils import calculate_accuracy, calculate_sensitivity_specificity, draw_roc_curve, draw_pr_curve


@torch.inference_mode()
def test_cls(net, dataloader, device, args, model_name, batch_size, amp, wandb=None):
    net.eval()
    num_val_batches = len(dataloader)
    true = 0
    record = np.zeros((2, args.classes))
    # iterate over the validation set
    pred_all = torch.zeros(0, device=device)
    gt_all = torch.zeros(0, device=device)
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, grade = batch[0], batch[1]

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            grade = grade.to(device=device, dtype=torch.long)

            # predict
            # softmax = torch.nn.Softmax(dim=1)
            # pred = softmax(net(image)).argmax(dim=1)
            pred = net(image)
            # for vgg, dim=0
            dim = 1
            # if model_name == 'vgg16':
            #     dim = 0
            print(pred, grade)
            true += (pred.argmax(dim=dim) == grade).sum()
            pred_all = torch.concat((pred_all, pred))
            gt_all = torch.concat((gt_all, grade))
    net.train()
    pred_all = pred_all.detach() if pred_all.requires_grad is True else pred_all
    gt_all = gt_all.detach() if gt_all.requires_grad is True else gt_all
    print(pred_all, gt_all)
    acc = calculate_accuracy(pred_all, gt_all, c=args.classes)
    metrics = calculate_sensitivity_specificity(pred_all, gt_all, c=args.classes)
    pr = draw_pr_curve(pred_all, gt_all, c=args.classes)
    if wandb is not None:
        wandb.log({"pr_curve": pr})
    rc = draw_roc_curve(pred_all, gt_all, c=args.classes)
    if wandb is not None:
        wandb.log({"roc_curve": rc})
    columns = ['accuracy', 'precision', 'f1_score', 'sensitivity', 'specificity']
    data = [acc, metrics['precision'], metrics['f1_score'], metrics['sensitivity'], metrics['specificity']]
    data = [[str(i) for i in data]]
    if wandb is not None:
        table = wandb.Table(data=data, columns=columns)
    if wandb is not None:
        wandb.log({"metrics table": table})
    return true / max(num_val_batches * batch_size, 1)


def get_args():
    parser = argparse.ArgumentParser(description='Train the Classification')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=3e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str,
                        default='/media/breeze/dev/Mf_Cls/checkpoints/classification/epochs[100]-bs[4]-lr[3e-05]-c2-ds[prostatex]-modal[adc]-focal/best_epoch100.pth',
                        help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='choose model from: resnet18, resnet34, resnet50,resnet101, vgg16, convnext, mfcls')
    parser.add_argument('--branch', type=int, default=2, help='denotes the number of streams')
    parser.add_argument('--branch_name', type=str, default='adc')
    parser.add_argument('--seed', type=int, default=12321)
    parser.add_argument('--aug', type=int, default=1, help='set aug equal to 1 to implement augmentation')
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--log', type=bool, default=True)
    parser.add_argument('--desc', type=str)
    return parser.parse_args()


from evaluate import evaluate_cls
import os
from util.data_loading import Cls_ProstateX_Dataset

if __name__ == '__main__':
    args = get_args()
    seed = args.seed
    dataset = Cls_ProstateX_Dataset(label_dir="/media/breeze/dev/Mf_Cls/data/ProstateX/predict_mask_deep/", num_classes=args.classes, branch_name=args.branch_name, test_mode=True)
    test_percent = 0.2
    n_test = int(len(dataset) * test_percent)
    n_train_val = len(dataset) - n_test
    train_val_set, test_set = random_split(dataset, [n_train_val, n_test],
                                           generator=torch.Generator().manual_seed(seed))
    loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)
    assert args.load is not None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_list = {'resnet18': Resnet_18(num_classes=args.classes),
                  'vgg16': VGG16()}
    logging.info(f'Using device {device}')
    assert args.model in model_list
    model = model_list[args.model]
    model = model.to(device)
    state_dict = torch.load(args.load, map_location=device)
    model.load_state_dict(state_dict)

    acc = test_cls(model, test_loader, device=device, args=args, model_name=args.model, batch_size=args.batch_size,
                   amp=args.amp)
    print("average accuracy is {}".format(acc))
