import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d_heatmap import UNet3D as Detector
from unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
import torch.nn.functional as F
import torch.nn as nn

from losses import DiceLoss
import utils

import gc
import torch

import nibabel as nib
import numpy as np

gc.collect()
torch.cuda.empty_cache()

def save_result(image, target, ground_truth, idx, save_dir='./results_voisegm'):
    # print('>> bef << ')
    # print('image.shape : ', image.shape)
    # print('target.shape : ', target.shape)
    # print('ground_truth.shape : ', ground_truth.shape)
    # print()

    image = image.squeeze(0).squeeze(0)
    target = target.squeeze(0).squeeze(0)
    ground_truth = ground_truth.squeeze(0).squeeze(0)


    np_image = image.detach().cpu().numpy()
    np_target = target.detach().cpu().numpy()
    np_gt = ground_truth.detach().cpu().numpy()

    image = np.swapaxes(image,0,2)
    target = np.swapaxes(target,0,2)
    ground_truth = np.swapaxes(ground_truth,0,2)

    nii_image = nib.Nifti1Image(np_image, affine=np.eye(4))
    nii_target = nib.Nifti1Image(np_target, affine=np.eye(4))
    nii_gt = nib.Nifti1Image(np_gt, affine=np.eye(4))

    nib.save(nii_image, save_dir + '/image_{}.nii.gz'.format(idx))
    nib.save(nii_target, save_dir + '/target_{}.nii.gz'.format(idx))
    nib.save(nii_gt, save_dir + '/gt_{}.nii.gz'.format(idx))


writer = SummaryWriter("runs")

detector = Detector(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES)
train_transforms = train_transform
val_transforms = val_transform

DETECTOR_WEIGHT_PATH = './checkpoints(voi detection)/epoch8_valLoss0.5606752634048462.pth'
# model = torch.load(MODEL_WEIGHT_PATH)
detector.load_state_dict(torch.load(DETECTOR_WEIGHT_PATH))

model = UNet3D(in_channels=1 , num_classes=1)
MODEL_WEIGHT_PATH = './checkpoints(voi segm)/epoch10_valLoss1.4958152770996094.pth'

if torch.cuda.is_available() and TRAIN_CUDA:
    detector = detector.cuda()
    model = model.cuda()
    # model = model.to('cuda')
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda 
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')


train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

# criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS)).cuda()

criterion = DiceLoss()
# criterion2 = nn.MSELoss()
# criterion3 = nn.L1Loss()

min_valid_loss = math.inf

detector.eval()
valid_losses = 0.0
model.eval()
with torch.no_grad():
    for idx, data in enumerate(val_dataloader):
        image, ground_truth = data['image'], data['label']

        heatmaps = detector(image)
        heatmaps = heatmaps.squeeze(0).detach().cpu().numpy()

        pred2kp = utils.get_maximum_point(heatmaps)
        pred2kp = pred2kp.numpy()
        pred2kp = pred2kp.astype(int)

        res = torch.zeros(1, 64, 128, 128)
        # res = np.zeros((64, 128, 128))

        for i in range(2):
            voi_start = pred2kp[2*i]
            voi_end = pred2kp[2*i + 1]
            voi = utils.voi_crop(image, [voi_start[0], voi_end[0]], [voi_start[1], voi_end[1]], [voi_start[2], voi_end[2]])
            voi = utils.resize_tensor(voi, (64, 128, 128))
            voi = voi.cuda()
            target = model(voi)
            res = utils.postprocess(target, res, [voi_start[0], voi_end[0]], [voi_start[1], voi_end[1]], [voi_start[2], voi_end[2]])

        res = res.cuda()

        valid_loss = criterion(res, ground_truth)
        print('Val Dice Score : ', valid_loss.item())
        valid_losses += valid_loss

        save_result(image, target, ground_truth, idx)


print()
print(f'=> Total Validation Score: {valid_losses / len(val_dataloader)}')
