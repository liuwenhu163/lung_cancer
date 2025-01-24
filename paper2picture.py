import torch
import torch.nn as nn
import sklearn

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BCEDiceLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        return 0.5 * bce + dice

def recall(input,target):
    recall = sklearn.metrics.recall_score(input,target)

    accuracy = torch.nn.functional.accuracy(input, target)
    f1 = sklearn.metrics.f1_score(input, target)

    return recall,accuracy,f1

import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def dice_coef(output, target):
    smooth = 1e-5

    # we need to use sigmoid because the output of Unet is logit.
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def dice_coef2(output, target):
    "This metric is for validation purpose"
    smooth = 1e-5

    output = output.view(-1)
    output = (output>0.5).float().cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

