import torch
import numpy as np


def multi_acc(pred, label):
    tags = torch.argmax(pred, dim=1)
    corrects = (tags == label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc


def multi_acc_class(pred, label, n_classes):
    accs_per_label_pct = []
    tags = torch.argmax(pred, dim=1)
    for cls in range(n_classes):
        corrects = label == cls
        num_total_per_label = corrects.sum()
        corrects &= tags == label
        num_corrects_per_label = corrects.float().sum()
        accs_per_label_pct.append(num_corrects_per_label / num_total_per_label * 100)
    return [i.item() for i in accs_per_label_pct]


def iou(outputs, labels, n_classes):
    ious = []
    pred = torch.argmax(outputs, dim=1).squeeze(1).view(-1)
    target = labels.view(-1)
    SMOOTH = 1e-6
    for idx in range(n_classes):
        preds_inds = pred == idx
        target_inds = target == idx
        intersection = (preds_inds & target_inds).float().sum()
        union = (preds_inds | target_inds).float().sum()
        iou = (intersection + SMOOTH) / (union + SMOOTH)
        ious.append(iou.item())
    return np.array(ious)


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def confusion_matrix_metrics(index, n_classes, conf_matrix):
    TP = conf_matrix.diag()
    idx = torch.ones(n_classes).byte()
    idx[index] = 0

    TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
    FP = conf_matrix[index, idx].sum()
    FN = conf_matrix[idx, index].sum()

    recall = TP[index] / (TP[index] + FN)
    precis = TP[index] / (TP[index] + FP)
    specificity = TN / (TN + FP)
    f1score = 2 * (precis * recall) / (precis + recall)

    return recall, precis, specificity, f1score
