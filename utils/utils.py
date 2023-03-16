import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def threshold_to_binary_class(tensor):
    # tensor: [9.9721e-01, 1.9753e-03, 8.2720e-07, 8.1001e-04]
    tensor_binary_class = torch.zeros(tensor.shape[0], device=tensor.device)
    for idx, i in enumerate(tensor):
        prob = 1 - i[0] - i[1]
        tensor_binary_class[idx] = prob
    # tensor = torch.where(tensor >= threshold, torch.tensor([1.], device=tensor.device),
    #                      torch.tensor([0.], device=tensor.device))
    return tensor_binary_class


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor):
    _, output = output.max(1)
    output = torch.where(output >= 2, torch.tensor([1.], device=output.device),
                         torch.tensor([0.], device=output.device))
    target = torch.where(target >= 2, torch.tensor([1.], device=target.device),
                         torch.tensor([0.], device=target.device))
    correct = output.eq(target).sum().item()
    total = target.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2f}")


def calculate_sensitivity_specificity(output, target):
    _, output = output.max(1)
    output = torch.where(output >= 2, torch.tensor([1.], device=output.device),
                         torch.tensor([0.], device=output.device))
    target = torch.where(target >= 2, torch.tensor([1.], device=target.device),
                         torch.tensor([0.], device=target.device))
    tn, fp, fn, tp = confusion_matrix(target.cpu().numpy(), output.cpu().numpy()).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")


# def calculate_auc(output, target):
#     _, output = output.max(1)
#     output = torch.where(output >= 2, torch.tensor([1.], device=output.device),
#                          torch.tensor([0.], device=output.device))
#     target = torch.where(target >= 2, torch.tensor([1.], device=target.device),
#                          torch.tensor([0.], device=target.device))
#     fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output.cpu().numpy())
#     roc_auc = auc(fpr, tpr)
#     print(f"AUC: {roc_auc:.2f}")


def draw_pr_curve(output, target):
    output = threshold_to_binary_class(output)
    target = torch.where(target >= 2, torch.tensor([1.], device=target.device),
                         torch.tensor([0.], device=target.device))
    precision, recall, _ = precision_recall_curve(target.cpu().numpy(), output.cpu().numpy())
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.show()


def draw_roc_curve(output, target):
    # Compute fpr, tpr, thresholds and AUC
    output = threshold_to_binary_class(output)
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
    plt.show()


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
