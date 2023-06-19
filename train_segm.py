import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset_segm import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transform_segm import (train_transform_cuda, val_transform_cuda)
import torch.nn.functional as F
import torch.nn as nn
# import torchvision.ops.focal_loss as FocalLoss
# from skimage.transform import resize

from losses import DiceLoss, FocalLoss, BinaryFocalLoss, DiceBCELoss
import utils

# from test_heatmap import save_result

import nibabel as nib
import numpy as np

import gc
import torch
from torch.optim.lr_scheduler import MultiStepLR

# from hourglass.HourGlassNet3D import HourglassNet
from vnet3d import HeatmapVNet

gc.collect()
torch.cuda.empty_cache()


writer = SummaryWriter("runs")

detector = HeatmapVNet()
MODEL_WEIGHT_PATH = './checkpoints/checkpoints (sigma2 + heatmaps)/epoch40_valLoss0.23936134576797485.pth'
detector.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
segm_model = UNet3D(in_channels=1 , num_classes=1)

detector = detector.cuda()
segm_model = segm_model.cuda()
train_transforms = train_transform_cuda
val_transforms = val_transform_cuda 


train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

# dice_criterion = DiceBCELoss()
# bce_criterion = nn.BCELoss()

# criterion2 = nn.MSELoss()
dice_criterion = BinaryFocalLoss()
# criterion3 = nn.L1Loss()

optimizer = Adam(params=segm_model.parameters(), lr=1e-3)
scheduler = MultiStepLR(optimizer, [24, 60, 120], gamma=0.1, last_epoch=-1)
# get_last_lr

from save_heatmap import save_result

min_valid_loss = math.inf
# alpha_pts = 1
# alpha_heatmap = 1

detector.eval()
for epoch in range(TRAINING_EPOCH):
    
    train_loss = 0.0
    segm_model.train()

    print('EPOCH : {} / {}, LR : {}, len(train_dataloader) : , '.format(epoch, TRAINING_EPOCH, optimizer.param_groups[0]["lr"]), len(train_dataloader))
    for idx, data in enumerate(train_dataloader):
        # if idx > 0:
        #     break
        
        '''
        image :  torch.Size([1, 1, 64, 128, 128])
        ground_truth :  torch.Size([1, 4, 64, 128, 128])
        target :  torch.Size([1, 4, 64, 128, 128])
        label_array :  torch.Size([1, 64, 128, 128])

        np_image :  (64, 128, 128)
        np_gts :  (4, 64, 128, 128)
        np_targets :  (4, 64, 128, 128)
        np_label :  (1, 64, 128, 128)

        '''


        image, ground_truth = data['image'], data['label']
        # target = model(image)
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
            # save_result(image, voi.squeeze(0), ground_truth, ground_truth, idx+i)
            target = segm_model(voi)

            crop_ground_truth = utils.voi_crop(ground_truth, [voi_start[0], voi_end[0]], [voi_start[1], voi_end[1]], [voi_start[2], voi_end[2]]).cuda()
            crop_ground_truth = utils.resize_tensor(crop_ground_truth, (64, 128, 128))
            # save_result(image, crop_ground_truth.squeeze(0), ground_truth, ground_truth, idx+i+10)
            
            loss = dice_criterion(target, crop_ground_truth)
            losses += loss

        # losses = loss_heatmap1 + loss_heatmap2 + loss_heatmap3

        # print(' {} / {} => Heatmap1 loss : {} | Heatmap2 loss : {} | Heatmap3 loss : {} | Total loss : {}'.format(
        #     idx+1, len(train_dataloader), loss_heatmap1.item(), loss_heatmap2.item(), loss_heatmap3.item(), losses.item()
        # ))
        print(' {} / {} => Total loss : {}'.format(idx+1, len(train_dataloader), losses.item()))

        losses.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_loss += losses.item()
    
    scheduler.step()

    valid_losses = 0.0
    # valid_dice_losses = 0.0

    segm_model.eval()
    with torch.no_grad():
        for val_idx, data in enumerate(val_dataloader):
            image, ground_truth = data['image'], data['label']
            
            # target = model(image)
            _, _, heatmap = detector(image)

            # if val_idx < 2:
            #     save_result(image, target, ground_truth, data['label_array'], val_idx)
            
            kps = utils.heatmap2kp(heatmap.squeeze(0).detach().cpu().numpy())
            kps = kps.numpy()
            kps = kps.astype(int)

            losses = 0.0
            # dice_losses = 0.0

            for i in range(2):
                voi_start = kps[2*i]
                voi_end = kps[2*i + 1]

                voi = utils.voi_crop(image, [voi_start[0], voi_end[0]], [voi_start[1], voi_end[1]], [voi_start[2], voi_end[2]]).cuda()
                # save_result(image, voi, ground_truth, ground_truth, idx+i)
                voi = utils.resize_tensor(voi, (64, 128, 128))
                target = segm_model(voi)

                crop_ground_truth = utils.voi_crop(ground_truth, [voi_start[0], voi_end[0]], [voi_start[1], voi_end[1]], [voi_start[2], voi_end[2]]).cuda()
                # save_result(image, crop_ground_truth, ground_truth, ground_truth, idx+i+10)
                crop_ground_truth = utils.resize_tensor(crop_ground_truth, (64, 128, 128))
                
                loss = dice_criterion(target, crop_ground_truth)
                losses += loss

                # dice_loss = dice_criterion(target, crop_ground_truth)
                # dice_losses += dice_loss

            valid_loss = losses
            valid_losses += valid_loss
            # valid_dice_losses += dice_losses

            
        writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
        writer.add_scalar("Loss/Validation", valid_losses / len(val_dataloader), epoch)
        # writer.add_scalar("Dice Loss/Validation", valid_dice_losses / len(val_dataloader), epoch)
        
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_losses / len(val_dataloader)}')
        # print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_losses / len(val_dataloader)} \t\t Validation Dice Loss : {valid_dice_losses / len(val_dataloader)}')
        
        valid_losses = valid_losses / len(val_dataloader)

        if min_valid_loss > valid_losses:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_losses:.6f}) \t Saving The Model')
            min_valid_loss = valid_losses
            # Saving State Dict
            torch.save(segm_model.state_dict(), f'checkpoints/checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')

writer.flush()
writer.close()