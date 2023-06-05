import torch
import torch.nn as nn
import torch.nn.functional as F

class Dice_avg(torch.nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self, bg):
        super(Dice_avg, self).__init__()
        self.bg = bg
        
    def forward(self, y_pred, y_true):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        y_true = y_true[:, self.bg:, :, :, :]
        y_pred = y_pred[:, self.bg:, :, :, :]
        top = 2 * ((y_true * y_pred).sum(dim=vol_axes))
        bottom = torch.clamp(((y_true + y_pred)).sum(dim=vol_axes), min=1e-5)
        dice = ((1-(top / bottom)))
        return torch.mean(dice), dice


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice