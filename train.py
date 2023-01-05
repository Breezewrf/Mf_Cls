import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from PIL import Image
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset, MSFDataset
from utils.dice_score import dice_loss
import cv2
from unetpp.unetpp_model import Nested_UNet
from loss import unet_loss, unetpp_loss
from msf_cls.msfusion import MSFusionNet
import random
import os

dir_t2w = './data/T2W_images/'
dir_adc = './data/ADC_images/'
dir_img = './data/T2W_images/'
dir_mask = './data/T2W_labels/'
dir_checkpoint = Path('./checkpoints/')
os.environ["WANDB_MODE"] = "offline"


def train_model(
        model,
        device,
        epochs: int = 2,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        save_interval: int = 10,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        branch: int = 1,
        seed=12321
):
    # 1. Create dataset
    dataset = None
    if branch == 1:
        try:
            dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
        except (AssertionError, RuntimeError):
            dataset = BasicDataset(dir_img, dir_mask, img_scale)
    elif branch == 2:
        dataset = MSFDataset(dir_t2w, dir_adc, dir_mask, img_scale)
    # 2. Split into train / validation partitions
    assert dataset is not None, f'the branch number is not set correctly: {branch}'
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

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
    
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must', settings=wandb.Settings(start_method="fork"))
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )
    wandb.run.name = "seed="+str(seed)+" lr="+str(learning_rate)
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=60)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                offset = 1 if model.name == 'msf' else 0
                if model.name == 'msf':
                    t2w_img, adc_img, true_masks = batch['t2w_image'], batch['adc_image'], batch['mask']
                    t2w_img = t2w_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    adc_img = adc_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    images = torch.stack((t2w_img, adc_img))
                    # if msf: branch x B x C x W x H
                else:
                    images, true_masks = batch['image'], batch['mask']
                    # if msf: branch x B x C x W x H
                    assert images.shape[1 + offset] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1 + offset]} channels. Please check that ' \
                        'the images are loaded correctly.'
                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                # images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.name == 'unet':
                        loss = unet_loss(model, masks_pred, true_masks)
                    elif model.name == 'unetpp':
                        loss = unetpp_loss(model, masks_pred, true_masks)
                        masks_pred = masks_pred[0]
                    else:
                        loss = unet_loss(model, masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0 + offset])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                if global_step % 50 == 0:
                    res = wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu())
                    res.image.show('output')
                    mask = wandb.Image(true_masks.float().cpu())
                    mask.image.show('gt')
                # cv2.waitKey(10000)
                # Evaluation round
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

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0 + offset].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                # **histograms
                            })
                            # res = wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu())
                            # res.image.show()
                        except:
                            pass

        if save_checkpoint and epoch % save_interval == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
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
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
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
    # try:
    #     train_model(
    #         model=model,
    #         epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         learning_rate=args.lr,
    #         device=device,
    #         img_scale=args.scale,
    #         val_percent=args.val / 100,
    #         amp=args.amp
    #     )
    # except torch.cuda.OutOfMemoryError:
    #     logging.error('Detected OutOfMemoryError! '
    #                   'Enabling checkpointing to reduce memory usage, but this slows down training. '
    #                   'Consider enabling AMP (--amp) for fast and memory efficient training')
    #     torch.cuda.empty_cache()
    #     model.use_checkpointing()
    #     train_model(
    #         model=model,
    #         epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         learning_rate=args.lr,
    #         device=device,
    #         img_scale=args.scale,
    #         val_percent=args.val / 100,
    #         amp=args.amp
    #     )
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
        seed=args.seed
    )
