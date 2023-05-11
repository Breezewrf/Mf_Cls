import torch
import torch.nn.functional as F
from tqdm import tqdm

from util.dice_score import multiclass_dice_coeff, dice_coeff
from loss import lw_loss
import numpy as np
import wandb
# wandb.init("experiments test")


def visualize_images(t2w_img, mask_true, mask_pred, name):
    # Convert images and masks to numpy arrays
    t2w_img = t2w_img.cpu().numpy().squeeze()
    mask_true = wandb.Image(mask_true.float().cpu())
    mask_pred = wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu())
    t2w_img = (t2w_img * 255).astype(np.uint8)

    t2w_img = np.stack((t2w_img,) * 3, axis=-1)

    # Visualize the images and masks
    # wandb.log({
    #     f"t2w_image": wandb.Image(t2w_img),
    #     f"true_mask": mask_true,
    #     f"predicted_mask": mask_pred
    # })


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, num_branch):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            if net.name == 'msf':
                assert num_branch in (2, 3)
                if num_branch == 2:
                    t2w_img, adc_img, mask_true = batch['t2w_image'], batch['adc_image'], batch['mask']
                    # move images and labels to correct device and type
                    t2w_img = t2w_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    adc_img = adc_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    image = torch.stack((t2w_img, adc_img))
                if num_branch == 3:
                    t2w_img, adc_img, dwi_img, mask_true = batch['t2w_image'], batch['adc_image'], batch['dwi_image'], batch['mask']
                    # move images and labels to correct device and type
                    t2w_img = t2w_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    adc_img = adc_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    dwi_img = dwi_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    image = torch.stack((t2w_img, adc_img, dwi_img))
            else:
                image, mask_true = batch['image'], batch['mask']

                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            if net.name == 'unetpp':
                mask_pred = mask_pred[0]
            if net.name == 'msf':
                visualize_images(t2w_img, mask_true, mask_pred, net.name)
            else:
                visualize_images(image, mask_true, mask_pred, net.name)
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)


@torch.inference_mode()
def evaluate_cls(net, dataloader, device, amp, model_name, batch_size):
    net.eval()
    num_val_batches = len(dataloader)
    true = 0

    # iterate over the validation set
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
            # if pred.argmax(dim=dim) == grade:
            #     true += 1
            # print("pred: ", pred.data, "\ngt: ", grade.data)
    net.train()
    return true / max(num_val_batches * batch_size, 1)
