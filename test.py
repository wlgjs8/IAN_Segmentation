from lib2to3.pgen2 import grammar
import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset import get_val_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from losses import DiceLoss
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
import torch.nn.functional as F
import torch.nn as nn

import nibabel as nib
import numpy as np

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1


def save_result(image, target, ground_truth, idx, save_dir='./results'):
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
    # target = target.transpose(2, 1, 0)
    # ground_truth = ground_truth.transpose(2, 1, 0)

    # print('>> aft << ')
    # print('image.shape : ', image.shape)
    # print('target.shape : ', target.shape)
    # print('ground_truth.shape : ', ground_truth.shape)
    # print()

    nii_image = nib.Nifti1Image(np_image, affine=np.eye(4))
    nii_target = nib.Nifti1Image(np_target, affine=np.eye(4))
    nii_gt = nib.Nifti1Image(np_gt, affine=np.eye(4))

    nib.save(nii_image, save_dir + '/image_{}.nii.gz'.format(idx))
    nib.save(nii_target, save_dir + '/target_{}.nii.gz'.format(idx))
    nib.save(nii_gt, save_dir + '/gt_{}.nii.gz'.format(idx))


model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES)

# MODEL_WEIGHT_PATH = './checkpoints#1/epoch4_valLoss0.7130498807546755.pth'
MODEL_WEIGHT_PATH = './checkpoints#2/epoch3_valLoss0.8230234489230827.pth'
# model = torch.load(MODEL_WEIGHT_PATH)
model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
# print(model)

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    # model = model.to('cuda')
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda 
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')


val_dataloader = get_val_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

criterion = DiceLoss()

min_valid_loss = math.inf
valid_score = 0.0
model.eval()

for idx, data in enumerate(val_dataloader):
    # if idx > 0:
    #     break
    image, ground_truth = data['image'], data['label']
    # image = image.float()
    # ground_truth = ground_truth.float()

    # print('image.shape : ', image.shape)
    # print('ground_truth.shape : ', ground_truth.shape)
    # print()

    target = model(image)
    

    save_result(image, target, ground_truth, idx)
    # print('image info : ', torch.max(image), ', ', torch.min(image))
    # print('target info : ', torch.max(target), ', ', torch.min(target))
    # print('ground_truth info : ', torch.max(ground_truth), ', ', torch.min(ground_truth))

    loss = criterion(target,ground_truth)
    dice_score = 1- loss.item()
    # valid_loss += loss.item()
    valid_score += dice_score

    # print('Val DICE Loss : ', loss)
    print('Val DICE Score : ', dice_score)
    # print('Val DICE Loss : ', valid_loss)
    
print()
print(f'=> Total Validation Score: {valid_score / len(val_dataloader)}')