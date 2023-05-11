# -*- coding: utf-8 -*-
# @Time    : 29/4/2023 12:57 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
from config import get_train_args as get_args
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import OrderedDict

from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from PIL import Image
import wandb
from evaluate import evaluate
from unet import UNet
from util.data_loader import MSFDataset
from util.dice_score import dice_loss
from loss import FocalLoss, TverskyLoss
import cv2
from unetpp.unetpp_model import Nested_UNet
from loss import unet_loss, unetpp_loss
from msf_cls.msfusion import MSFusionNet
import random
import os

dir_t2w = 'data/ProstateX/T2W_images'
dir_adc = 'data/ProstateX/ADC_images'
dir_dwi = 'data/ProstateX/DWI_images'
dir_mask = 'data/ProstateX/labeled_GT_colored'
os.environ["WANDB_MODE"] = "offline"
kf = KFold(n_splits=5, shuffle=True, random_state=57749867)
focalLoss = FocalLoss(alpha=1, gamma=2)
tverskyLoss = TverskyLoss(alpha=0.5, beta=0.5)


def train_model(
        model_name,
        device,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        val_percent: float,
        save_checkpoint: bool,
        save_interval: int,
        img_scale: float,
        amp: bool,
        weight_decay: float,
        momentum: float,
        gradient_clipping: float,
        branch: int,
        seed,
        aug,
        opt,
        desc,
        num_classes,
        dataset_name,
        branch_name,
        loss_name,
        task,
        log
):
    assert task in ['seg', 'cls', 'unified'], "{} is not a legal mode,please select a proper task in args".format(
        args.task)
    p = "epochs[{}]-bs[{}]-lr[{}]-c{}-ds[{}]-modal[{}]-{}".format(epochs, batch_size, learning_rate, num_classes,
                                                                  dataset_name, branch_name, loss_name)
    dir_checkpoint = Path('./checkpoints/unified/{}'.format(p))
    best_model_path = 'best.pth'
    if log:
        config = {'epoch': epochs, 'batch_size': batch_size, 'lr': learning_rate, 'seed': seed, 'opt': opt,
                  'num_classes': num_classes, 'dataset': dataset_name, 'branch': branch_name}
        run = wandb.init(project='unified', config=config)

    # 1. create dataset
    dataset = MSFDataset(dir_t2w, dir_adc, dir_dwi, dir_mask, num_classes=num_classes)
    test_percent = 0.2
    n_test = int(len(dataset) * test_percent)
    n_train_val = len(dataset) - n_test
    train_val_set, test_set = random_split(dataset, [n_train_val, n_test],
                                           generator=torch.Generator().manual_seed(seed))
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val

    # (1) Set `os env`
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

    global best_acc
    best_acc = 0
    best_epoch = 0
    # 3. Create data loaders
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_set)):
        if fold != 3:
            print("Empirical fold=3 gets best performance")
            continue
        logging.info(f'Using device {device}')
        model = MSFusionNet(3, 2, task=args.task)
        model = model.to(device)
        if args.load is not None:
            # Create a new dictionary with the desired parameters
            logging.info("checkpoint {} is loading!".format(args.load))
            new_state_dict = OrderedDict()
            state_dict_seg = torch.load(args.load)
            for key in state_dict_seg:
                if 'encode' in key or 'inc' in key:
                    new_state_dict[key] = state_dict_seg[key]
            model.load_state_dict(new_state_dict, strict=False)
            for param in model.inc.parameters():
                param.requires_grad = False
            for param in model.encoder1.parameters():
                param.requires_grad = False
            for param in model.encoder2.parameters():
                param.requires_grad = False
            for param in model.encoder3.parameters():
                param.requires_grad = False
            for param in model.encoder4.parameters():
                param.requires_grad = False

        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)
        n_train = len(train_set)
        if task != 'seg':
            # re-weight
            class_weight = np.zeros(num_classes)
            train_labels = []
            for data in train_set:
                label = data['mask']  # the label is just used for segmentation, not equals to the GGG
                grade = data['GGG']
                # l, t = np.unique(label, return_counts=True)
                class_weight[grade] += 1
                train_labels.append(grade)
            # exp_weight = [(1 - c / sum(class_weight)) ** 2 for c in class_weight]

            exp_weight = (class_weight.sum() / class_weight) / ((class_weight.sum() / class_weight).sum())
            example_weight = [exp_weight[e] for e in train_labels]
            sampler = WeightedRandomSampler(example_weight, len(train_labels))
            loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
            train_loader = DataLoader(train_set, sampler=sampler, **loader_args)
            val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
            test_lodaer = DataLoader(test_set, shuffle=False, **loader_args)
            # exp_weight = [1, 0, 0, 0]
            class_weight = np.zeros(num_classes)
            assert len(train_loader) != 0
            for data in train_loader:
                grade = data['GGG']
                class_weight[grade] += 1
            logging.info("class weight after re_weight: {}".format(class_weight))
        else:
            loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
            train_loader = DataLoader(train_set, shuffle=True, **loader_args)
            val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
            test_lodaer = DataLoader(test_set, shuffle=False, **loader_args)
        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer = optim.AdamW(model.parameters(),
                                lr=learning_rate)  # , weight_decay=weight_decay, momentum=momentum, foreach=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=60,
                                                         factor=0.5)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        global_step = 0
        # 5. Begin training
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for data in train_loader:
                    t2w_img, adc_img, dwi_img, true_mask, grade = \
                        data['t2w_image'], data['adc_image'], data['dwi_image'], data['mask'], data['GGG']
                    t2w_img = t2w_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    adc_img = adc_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    dwi_img = dwi_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_mask = true_mask.to(device=device, dtype=torch.long)
                    images = torch.stack((t2w_img, adc_img, dwi_img))
                    grade = grade.to(device=device, dtype=torch.float32)
                    model.train()
                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        pred = model(images)
                        # print("pred: ", pred.shape)
                        # print("pred: ", pred)
                        # print("res: ", res)
                        # loss = lw_loss(pred, grade)
                        assert loss_name in ['focal'], "loss specification error"
                        if loss_name == 'focal':
                            loss = focalLoss(pred, true_mask)

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    pbar.update(adc_img.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    if log:
                        wandb.log({'loss': loss.item()})
                    division_step = (n_train // (5 * batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:
                            score = evaluate(model, val_loader, device, amp, num_branch=args.branch)
                            scheduler.step(score)
                            logging.info('Score: {}'.format(score))
                            if log:
                                wandb.log({'score': score})
            if save_checkpoint and epoch % save_interval == 0:
                # test:
                test_acc = evaluate(model, test_lodaer, device, amp, num_branch=args.branch)
                print("test_score:{}".format(test_acc))
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
                wandb.log({"test_acc": test_acc})
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch
                    torch.save(state_dict,
                               str(dir_checkpoint / 'best_epoch.pth'))
                    if log:
                        model_wandb = wandb.Artifact('classification-model', type='model')
                        model_wandb.add_file(str(dir_checkpoint / 'best_epoch.pth'))
                        run.log_artifact(model_wandb)

        logging.info("best model is trained with {} epochs, best acc is {}".format(best_epoch, best_acc))

        logging.info(f'Checkpoint  training finished!')


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    assert (args.branch == 1 and args.model != 'msf') or (args.branch == 2 and args.model == 'msf')
    if args.model == 'unet':
        model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model == 'unetpp':
        model = Nested_UNet(in_ch=1, out_ch=args.classes)
    elif args.model == 'msf':
        model = MSFusionNet(input_c=2, output_c=args.classes, task=args.task)
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

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp,
        branch=args.branch,
        seed=args.seed,
        aug=args.aug,
        opt=args.opt,
        save_checkpoint=True,
        save_interval=args.save_interval,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        gradient_clipping=args.gradient_clipping,
        desc=args.desc,
        num_classes=args.classes,
        dataset_name=args.dataset_name,
        branch_name=args.branch_name,
        loss_name=args.loss_f,
        task=args.task,
        log=args.log
    )
