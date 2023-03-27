import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np
import cv2
from PIL import Image


def imshow(img, desc=""):
    # Check if input is a numpy array
    if isinstance(img, np.ndarray):
        # Check if input is a floating point tensor
        if img.dtype == np.float32 or img.dtype == np.float64:
            # Scale the image to the range [0, 1]
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            # Convert the image to a 8 bits unsigned integer
            img = (img * 255).astype(np.uint8)
        # Check if input is a grayscale image
        if len(img.shape) == 2:
            # Convert the image to a color image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Check if input is a PyTorch tensor
    elif torch.is_tensor(img):
        # Convert the tensor to a numpy array
        img = img.detach().cpu().numpy()
        # Scale the image to the range [0, 1]
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        # Convert the image to a 8 bits unsigned integer
        img = (img * 255).astype(np.uint8)
        # Check if input is a grayscale image
        if len(img.shape) == 2:
            # Convert the image to a color image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Check if input is an OpenCV image
    elif isinstance(img, np.ndarray):
        # Convert the image to a numpy array
        img = np.asarray(img)
    # Check if input is a PIL image
    elif isinstance(img, Image.Image):
        # Convert the image to a numpy array
        img = np.array(img)
    # Check if input is a list of images
    elif isinstance(img, list):
        # Concatenate the images horizontally
        img = np.concatenate(img, axis=1)
    # Check if input is a tuple of images
    elif isinstance(img, tuple):
        # Concatenate the images horizontally
        img = np.concatenate(img, axis=1)
    # Show the image using matplotlib
    plt.imshow(img)
    plt.title = desc
    plt.text = desc
    plt.show()


def threshold_to_binary_class(tensor, c=2):
    # tensor: [9.9721e-01, 1.9753e-03, 8.2720e-07, 8.1001e-04]
    tensor_binary_class = torch.zeros(tensor.shape[0], device=tensor.device)
    for idx, i in enumerate(tensor):
        prob = 1 - i[0]
        if c == 4:
            prob = 1 - i[0] - i[1]
        tensor_binary_class[idx] = prob
    # tensor = torch.where(tensor >= threshold, torch.tensor([1.], device=tensor.device),
    #                      torch.tensor([0.], device=tensor.device))
    return tensor_binary_class


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor, c=2):
    _, output = output.max(1)
    if c == 4:
        output = torch.where(output >= 2, torch.tensor([1.], device=output.device),
                             torch.tensor([0.], device=output.device))
        target = torch.where(target >= 2, torch.tensor([1.], device=target.device),
                             torch.tensor([0.], device=target.device))
    correct = output.eq(target).sum().item()
    total = target.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2f}")
    return accuracy


def calculate_sensitivity_specificity(output, target, c=2):
    _, output = output.max(1)
    if c == 4:
        output = torch.where(output >= 2, torch.tensor([1.], device=output.device),
                             torch.tensor([0.], device=output.device))
        target = torch.where(target >= 2, torch.tensor([1.], device=target.device),
                             torch.tensor([0.], device=target.device))
    tn, fp, fn, tp = confusion_matrix(target.cpu().numpy(), output.cpu().numpy()).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)  # identical to sensitivity
    f1_score = 2 * precision * recall / (precision + recall)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"f1_score: {f1_score:.2f}")
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'sensitivity': sensitivity,
            'specificity': specificity}


# def calculate_auc(output, target):
#     _, output = output.max(1)
#     output = torch.where(output >= 2, torch.tensor([1.], device=output.device),
#                          torch.tensor([0.], device=output.device))
#     target = torch.where(target >= 2, torch.tensor([1.], device=target.device),
#                          torch.tensor([0.], device=target.device))
#     fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output.cpu().numpy())
#     roc_auc = auc(fpr, tpr)
#     print(f"AUC: {roc_auc:.2f}")


def draw_pr_curve(output, target, c=2):
    output = threshold_to_binary_class(output, c=c)
    if c == 4:
        target = torch.where(target >= 2, torch.tensor([1.], device=target.device),
                             torch.tensor([0.], device=target.device))
    precision, recall, _ = precision_recall_curve(target.cpu().numpy(), output.cpu().numpy())
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    # plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    # plt.show()
    return plt


def draw_roc_curve(output, target, c=2):
    # Compute fpr, tpr, thresholds and AUC
    output = threshold_to_binary_class(output, c=c)
    if c == 4:
        target = torch.where(target >= 2, torch.tensor([1.], device=target.device),
                             torch.tensor([0.], device=target.device))
    fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output.detach().cpu().numpy())
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    # plt.show()
    return plt


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.show()
