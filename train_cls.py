# -*- coding: utf-8 -*-
# @Time    : 15/2/2023 12:18 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
import argparse
import logging
import torch
from msf_cls.backbone.resnet import resnet34, resnet18, resnet50, resnet101
from msf_cls.backbone.convnext import ConvNeXt
from utils.data_loading import Cls_Dataset
import os
import random
from torch.utils.data import DataLoader, random_split
import numpy as np
from torch import optim
from tqdm import tqdm
from evaluate import evaluate_cls
from pathlib import Path

dir_checkpoint = Path('./checkpoints/classification/')
from loss import lw_loss


def train_model(
        model,
        device,
        epochs: int = 2,
        batch_size: int = 1,
        learning_rate: float = 3e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        save_interval: int = 50,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        branch: int = 1,
        seed=12321,
        aug=1,
        opt='adamw',
        desc=''
):
    # 1. create dataset
    dataset = Cls_Dataset()
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
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

    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

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
            for img, grade in train_loader:
                img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                grade = grade.to(device=device, dtype=torch.float32)
                model.train()
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    pred = model(img)
                    # print("pred: ", pred.shape)
                    # print("pred: ", pred)
                    # print("res: ", res)
                    loss = lw_loss(pred, grade)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(img.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in model.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     if not torch.isinf(value).any():
                        #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     if not torch.isinf(value.grad).any():
                        #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        score = evaluate_cls(model, val_loader, device, amp)
                        scheduler.step(score)
                        logging.info('Score: {}'.format(score))
        if save_checkpoint and epoch % save_interval == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    Path('epoch' + str(epochs)).mkdir(parents=True, exist_ok=True)
    state_dict_final = model.state_dict()
    torch.save(state_dict_final,
               str(Path('epoch' + str(epochs)) / '{}_final.pth'.format(desc)))
    logging.info(f'Checkpoint  training finished!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the Classification')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=3e-7,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--model', type=str, default='msf', help='choose model from: unet, unetpp, msfunet, mfcls')
    parser.add_argument('--branch', type=int, default=2, help='denotes the number of streams')
    parser.add_argument('--seed', type=int, default=12321)
    parser.add_argument('--aug', type=int, default=1, help='set aug equal to 1 to implement augmentation')
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--desc', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    model = resnet18()
    model = model.to(device)
    train_model(
        model=model,
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
        desc=args.desc
    )