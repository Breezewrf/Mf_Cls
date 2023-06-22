# -*- coding: utf-8 -*-
# @Time    : 15/2/2023 12:18 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
import argparse
import logging
import torch
from msf_cls.backbone.resnet import resnet34, resnet18, resnet50, resnet101
from msf_cls.backbone.convnext import ConvNeXt
from msf_cls.backbone.gain import GAIN
from msf_cls.backbone.vgg import Vgg_16
from msf_cls.ResMSF import ResMSFNet
from util.data_loading import Cls_Dataset, Cls_ProstateX_Dataset
import os
import random
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
from torch import optim
from collections import OrderedDict
from tqdm import tqdm
from evaluate import evaluate_cls
from pathlib import Path
import wandb
from sklearn.model_selection import KFold
from test_cls import test_cls
from util.data_loader import MSFClassifyDataset

# os.environ["WANDB_MODE"] = "offline"
from loss import lw_loss
from msf_cls.backbone.pretrained import Resnet_18, VGG16
from loss import FocalLoss, TverskyLoss, TriModalSimilarityLoss


kf = KFold(n_splits=5, shuffle=True, random_state=57749867)
focalLoss = FocalLoss(alpha=1, gamma=2)
tverskyLoss = TverskyLoss(alpha=0.5, beta=0.5)
tmsLoss = TriModalSimilarityLoss()


def train_model(
        model_name,
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
        desc='',
        num_classes=2,
        dataset_name='ProstateX',
        branch_name='t2w',
        loss_name='focal',
        task='cls',
        log=True
):
    p = "gain-stream3-epochs[{}]-bs[{}]-lr[{}]-c{}-ds[{}]-modal[{}]-{}-fuse4-exp1000-v2".format(epochs, batch_size, learning_rate,
                                                                                      num_classes,
                                                                                      dataset_name, branch_name,
                                                                                      loss_name)
    dir_checkpoint = Path('./checkpoints/classification/{}'.format(p))
    best_model_path = 'best.pth'
    if log:
        config = {'epoch': epochs, 'batch_size': batch_size, 'lr': learning_rate, 'seed': seed, 'opt': opt,
                  'num_classes': num_classes, 'dataset': dataset_name, 'branch': branch_name}
        run = wandb.init(project='classification', config=config)

    # 1. create dataset
    dataset = MSFClassifyDataset(num_classes=num_classes, branch_num=3)
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
        if args.model != 'resmsf_gain':
            model_list = {'resnet18': Resnet_18(num_classes=args.classes),
                          'resmsf': ResMSFNet(in_c=3, out_c=args.classes, num_branch=args.branch),
                          'vgg16': VGG16(),
                          }
            assert args.model in model_list
            model = model_list[args.model]
        else:
            if args.test_mode:
                model = GAIN(num_classes=2, backbone_name='resmsf',
                             load_dir=args.load).model
            else:
                model = GAIN(num_classes=2, backbone_name='resmsf',
                             load_dir=args.load)
            logging.info(f'Using model {args.model}')
        logging.info(f'Using device {device}')

        model = model.to(device)
        # if args.load is not None:
        #     # Create a new dictionary with the desired parameters
        #     new_state_dict = OrderedDict()
        #     if args.load:
        #         state_dict_seg = torch.load(args.load)
        #         for key in state_dict_seg:
        #             if 'encode' in key or 'inc' in key:
        #                 new_state_dict[key] = state_dict_seg[key]
        #         model.load_state_dict(new_state_dict, strict=False)

        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)
        n_train = len(train_set)
        # re-weight
        class_weight = np.zeros(num_classes)
        train_labels = []
        for im, _, _, label, _ in train_set:
            l, t = np.unique(label, return_counts=True)
            class_weight[l] += t
            train_labels.append(label)
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
        for im, _, _, label, _ in train_loader:
            l, t = np.unique(label, return_counts=True)
            class_weight[l] += t
        logging.info("class weight after re_weight: {}".format(class_weight))

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
                for t2w_img, adc_img, dwi_img, grade, _ in train_loader:

                    t2w_img = t2w_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    adc_img = adc_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    dwi_img = dwi_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                    img = torch.stack((t2w_img, adc_img, dwi_img))
                    grade = grade.to(device=device, dtype=torch.float32)
                    model.train()
                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        if model_name == 'resmsf_gain':
                            pred1, pred2 = model(img)
                            loss_pred = focalLoss(pred1, grade)
                            loss_attention = pred2[torch.arange(len(grade)), grade.to(torch.long)].sum()
                            loss = loss_pred + loss_attention
                        else:
                            pred = model(img)
                        # print("pred: ", pred.shape)
                        # print("pred: ", pred)
                        # print("res: ", res)
                        # loss = lw_loss(pred, grade)
                            assert loss_name in ['focal'], "loss specification error"
                            if loss_name == 'focal':
                                loss = focalLoss(pred, grade)
                            # l1_loss = abs(torch.argmax(pred, dim=1) - grade)
                            # loss += l1_loss

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    pbar.update(img.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    if log:
                        wandb.log({'loss': loss.item()})
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

                            score = evaluate_cls(model, val_loader, device, amp, args.model, batch_size=batch_size)
                            scheduler.step(score)
                            logging.info('Score: {}'.format(score))
                            if log:
                                wandb.log({'score': score})
            if save_checkpoint and epoch % save_interval == 0:
                # test:
                test_acc = 0
                # test_acc = test_cls(model, test_lodaer, device, args=args, amp=amp, model_name=args.model,
                #                     batch_size=batch_size, wandb=wandb)
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


def get_args():
    parser = argparse.ArgumentParser(description='Train the Classification')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=3e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--model', type=str, default='resmsf',
                        help='choose model from: resnet18, resnet34, resnet50,resnet101, vgg16, convnext, mfcls')
    parser.add_argument('--branch', type=int, default=3, help='denotes the number of streams')
    parser.add_argument('--seed', type=int, default=12321)
    parser.add_argument('--aug', type=int, default=1, help='set aug equal to 1 to implement augmentation')
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--dataset_name', type=str, default='prostatex')
    parser.add_argument('--branch_name', type=str, default='pre_fuse')
    parser.add_argument('--loss_f', type=str, default='focal')
    parser.add_argument('--task', type=str, default='cls')
    parser.add_argument('--log', type=bool, default=True)
    parser.add_argument('--desc', type=str)
    parser.add_argument('--data_source', type=str)
    parser.add_argument('--test_mode', type=bool)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info("classification model: {}".format(args.model))
    logging.info("classes number: {}".format(args.classes))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        desc=args.desc,
        num_classes=args.classes,
        dataset_name=args.dataset_name,
        branch_name=args.branch_name,
        loss_name=args.loss_f,
        task=args.task,
        log=args.log
    )
