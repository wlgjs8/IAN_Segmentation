import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset_segm import get_val_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transform_segm import (train_transform_cuda, val_transform_cuda)
import torch.nn.functional as F
import torch.nn as nn

from losses import DiceLoss, FocalLoss, BinaryFocalLoss, DiceBCELoss
import utils

import nibabel as nib
import numpy as np

import gc
import torch

from config import (
    RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH
)

from vnet3d import HeatmapVNet

gc.collect()
torch.cuda.empty_cache()

detector = HeatmapVNet()
DETECTOR_WEIGHT_PATH = './checkpoints/checkpoints (sigma2 + heatmaps)/epoch40_valLoss0.23936134576797485.pth'
detector.load_state_dict(torch.load(DETECTOR_WEIGHT_PATH))

segm_model = UNet3D(in_channels=1 , num_classes=1)
# MODEL_WEIGHT_PATH = './checkpoints/checkpoints(bce)/epoch21_valLoss0.022542281076312065.pth'
MODEL_WEIGHT_PATH = './checkpoints/checkpoints(segm focal2)/epoch9_valLoss0.008554019965231419.pth'
segm_model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))

detector = detector.cuda()
segm_model = segm_model.cuda()
train_transforms = train_transform_cuda
val_transforms = val_transform_cuda 

val_dataloader = get_val_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

dice_criterion = DiceLoss()

def save_result(image, gt, crop_image, crop_prediction, crop_gt, idx=0, save_dir='./results/results_voisegm2'):
    '''
    Image
    Crop Image
    Crop Prediction
    GT
    Crop GT

    # Whole Prediction

    image :  torch.Size([1, 1, 64, 128, 128])
    gt :  torch.Size([1, 1, 64, 128, 128])
    crop_image :  torch.Size([1, 1, 64, 128, 128])
    crop_gt :  torch.Size([1, 1, 64, 128, 128])
    '''

    image = image.squeeze(0).squeeze(0).detach().cpu().numpy()
    gt = gt.squeeze(0).squeeze(0).detach().cpu().numpy()
    crop_image = crop_image.squeeze(0).squeeze(0).detach().cpu().numpy()
    crop_prediction = crop_prediction.squeeze(0).squeeze(0).detach().cpu().numpy()
    crop_gt = crop_gt.squeeze(0).squeeze(0).detach().cpu().numpy()

    nii_image = nib.Nifti1Image(image, affine=np.eye(4))
    nii_gt = nib.Nifti1Image(gt, affine=np.eye(4))
    nii_crop_image = nib.Nifti1Image(crop_image, affine=np.eye(4))
    nii_crop_prediction = nib.Nifti1Image(crop_prediction, affine=np.eye(4))
    nii_crop_gt = nib.Nifti1Image(crop_gt, affine=np.eye(4))

    nib.save(nii_image, save_dir + '/image_{}.nii.gz'.format(idx))
    nib.save(nii_gt, save_dir + '/gt_{}.nii.gz'.format(idx))
    nib.save(nii_crop_image, save_dir + '/crop_image_{}.nii.gz'.format(idx))
    nib.save(nii_crop_prediction, save_dir + '/crop_prediction_{}.nii.gz'.format(idx))
    nib.save(nii_crop_gt, save_dir + '/crop_gt_{}.nii.gz'.format(idx))


min_valid_loss = math.inf
detector.eval()
segm_model.eval()

with torch.no_grad():
    for val_idx, data in enumerate(val_dataloader):
        if val_idx > 0:
            break

        image, ground_truth = data['image'], data['label']
        
        _, _, heatmap = detector(image)

        kps = utils.heatmap2kp(heatmap.squeeze(0).detach().cpu().numpy())
        kps = kps.numpy()
        kps = kps.astype(int)

        losses = 0.0

        for i in range(2):
            voi_start = kps[2*i]
            voi_end = kps[2*i + 1]

            voi = utils.voi_crop(image, [voi_start[0], voi_end[0]], [voi_start[1], voi_end[1]], [voi_start[2], voi_end[2]]).cuda()
            voi = utils.resize_tensor(voi, (64, 128, 128))
            target = segm_model(voi)

            # target = (target>torch.mean(target)).float()
            target = (target>0.2).float()

            crop_ground_truth = utils.voi_crop(ground_truth, [voi_start[0], voi_end[0]], [voi_start[1], voi_end[1]], [voi_start[2], voi_end[2]]).cuda()
            crop_ground_truth = utils.resize_tensor(crop_ground_truth, (64, 128, 128))
            
            save_result(image, ground_truth, voi, target, crop_ground_truth, val_idx*2 + i)

            loss = dice_criterion(target, crop_ground_truth)
            losses += loss

        print(f'Val {val_idx+1} / {len(val_dataloader)} => Dice Loss : {losses.item()}')
