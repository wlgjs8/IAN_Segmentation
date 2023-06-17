from lib2to3.pgen2 import grammar
import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset_heatmap import get_val_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# from unet3d import UNet3D
# from unet3d_heatmap import UNet3D
from vnet3d import HeatmapVNet

from losses import DiceLoss
from transforms import (train_transform_cuda, val_transform_cuda)
import torch.nn.functional as F
import torch.nn as nn

import nibabel as nib
import numpy as np

import utils
from config import (
    RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH
)

def save_result(image, target, ground_truth, label_array, idx, save_dir='./results/results focal sigma2'):
    if len(image.shape) == 3:
        np_image = image.detach().cpu().numpy()
    else:
        np_image = image.squeeze(0).squeeze(0).detach().cpu().numpy()
    
    target= target.squeeze(0)
    
    # target= target.squeeze(0).squeeze(0)
    # np_target= target.detach().cpu().numpy()
    np_target, _ = torch.max(target, axis=0)
    np_target= np_target.detach().cpu().numpy()

    # ground_truth= ground_truth.squeeze(0).squeeze(0)
    # np_ground_truth= ground_truth.detach().cpu().numpy()

    ground_truth = ground_truth.squeeze(0)
    np_ground_truth, _ = torch.max(ground_truth, axis=0)
    np_ground_truth= np_ground_truth.detach().cpu().numpy()

    kp_arr = np.zeros((RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH))
    gtkp_arr = np.zeros((RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH))
    # print('target ; ', target.shape)
    target2kp = utils.heatmap2kp(target.detach().cpu().numpy())
    target2kp = target2kp.numpy()
    target2kp = target2kp.astype(int)

    gt_points = utils.heatmap2kp(ground_truth.squeeze(0).detach().cpu().numpy())
    gt_points = gt_points.numpy()
    gt_points = gt_points.astype(int)

    for gtkp in gt_points:
        gtkp_arr[gtkp[0]][gtkp[1]][gtkp[2]] = 1

    for kp in target2kp:
        kp_arr[kp[0]][kp[1]][kp[2]] = 1

    nii_image = nib.Nifti1Image(np_image, affine=np.eye(4))
    nii_target = nib.Nifti1Image(np_target, affine=np.eye(4))
    nii_ground_truth = nib.Nifti1Image(np_ground_truth, affine=np.eye(4))
    nii_kp = nib.Nifti1Image(kp_arr, affine=np.eye(4))
    nii_gt_kp = nib.Nifti1Image(gtkp_arr, affine=np.eye(4))
    nii_label_array = nib.Nifti1Image(label_array.squeeze(0).detach().cpu().numpy(), affine=np.eye(4))

    nib.save(nii_image, save_dir + '/image_{}.nii.gz'.format(idx))
    nib.save(nii_target, save_dir + '/heatmap_{}.nii.gz'.format(idx))
    nib.save(nii_ground_truth, save_dir + '/gt_heatmap_{}.nii.gz'.format(idx))
    nib.save(nii_kp, save_dir + '/keypoints_{}.nii.gz'.format(idx))
    nib.save(nii_gt_kp, save_dir + '/gt_keypoints_{}.nii.gz'.format(idx))
    nib.save(nii_label_array, save_dir + '/gt_dense_{}.nii.gz'.format(idx))

model = HeatmapVNet()

MODEL_WEIGHT_PATH = './checkpoints/checkpoints/epoch40_valLoss0.23936134576797485.pth'
model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))

model = model.cuda()
train_transforms = train_transform_cuda
val_transforms = val_transform_cuda 

val_dataloader = get_val_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

model.eval()

for idx, data in enumerate(val_dataloader):
    # if idx > 1:
    #     break
    image, ground_truth = data['image'], data['label']

    # target = model(image)
    _, _, target = model(image)

    '''
    print('image : ', image.shape)
    print('ground_truth : ', ground_truth.shape)
    print('target : ', target.shape)

    image :  torch.Size([1, 1, 64, 128, 128])
    ground_truth :  torch.Size([1, 1, 4, 64, 128, 128])
    target :  torch.Size([1, 4, 64, 128, 128])
    '''

    # ground_truth = ground_truth.squeeze(0)
    save_result(image, target, ground_truth, data['label_array'], idx)