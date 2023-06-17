import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset_heatmap import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d_heatmap import UNet3D
from transforms import (train_transform_cuda, val_transform_cuda)
import torch.nn.functional as F
import torch.nn as nn
# import torchvision.ops.focal_loss as FocalLoss
# from skimage.transform import resize

from losses import DiceLoss, FocalLoss, BinaryFocalLoss
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

# model = UNet3D(in_channels=960 , num_classes= 4)
model = HeatmapVNet()

model = model.cuda()
train_transforms = train_transform_cuda
val_transforms = val_transform_cuda 


train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

# criterion = DiceLoss()

# criterion2 = nn.MSELoss()
criterion2 = BinaryFocalLoss()
# criterion3 = nn.L1Loss()

optimizer = Adam(params=model.parameters(), lr=1e-3)
scheduler = MultiStepLR(optimizer, [24, 60, 120], gamma=0.1, last_epoch=-1)
# get_last_lr

from save_heatmap import save_result

min_valid_loss = math.inf
# alpha_pts = 1
# alpha_heatmap = 1

for epoch in range(TRAINING_EPOCH):
    
    train_loss = 0.0
    model.train()

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
        target1, target2, target3 = model(image)

        # print('ground_truth : ', ground_truth.shape)

        ground_truth1 = utils.resize_gt(ground_truth, (1, 4, 16, 32, 32))
        ground_truth2 = utils.resize_gt(ground_truth, (1, 4, 32, 64, 64))
        ground_truth3 = ground_truth

        

        '''
        target1 :  torch.Size([1, 128, 16, 32, 32])
        target2 :  torch.Size([1, 64, 32, 64, 64])
        target3 :  torch.Size([1, 4, 64, 128, 128])

        print('target1 : ', target1.shape)
        print('target2 : ', target2.shape)
        print('target3 : ', target3.shape)
        print()
        '''

        # print('image : ', image.shape)
        # print('target : ', target.shape)

        '''
        save_dir='./results/results_reproduce'

        np_image = image.squeeze(0).squeeze(0).detach().cpu().numpy()
        np_gts = ground_truth.squeeze(0).detach().cpu().numpy()
        np_targets = target.squeeze(0).detach().cpu().numpy()
        np_label = data['label_array'].squeeze(0).detach().cpu().numpy()

        nii_image = nib.Nifti1Image(np_image, affine=np.eye(4))
        nii_label = nib.Nifti1Image(np_label, affine=np.eye(4))

        nib.save(nii_image, save_dir + '/image.nii.gz')
        nib.save(nii_label, save_dir + '/label.nii.gz')

        for i in range(4):
            nii_gt = nib.Nifti1Image(np_gts[i,:,:,:], affine=np.eye(4))
            nib.save(nii_gt, save_dir + '/gt_{}.nii.gz'.format(i))

            nii_target = nib.Nifti1Image(np_targets[i, :,:,:], affine=np.eye(4))
            nib.save(nii_target, save_dir + '/target_{}.nii.gz'.format(i))

        tensor_kpfromgt = utils.heatmap2kp(np_gts)
        tensor_kpfromgt = tensor_kpfromgt.numpy()
        tensor_kpfromgt = tensor_kpfromgt.astype(int)

        np_kpfromgt = np.zeros((64, 128, 128))
        for i in range(4):
            np_kpfromgt[tensor_kpfromgt[i][0]][tensor_kpfromgt[i][1]][tensor_kpfromgt[i][2]] = 1

        nii_kpgt = nib.Nifti1Image(np_kpfromgt, affine=np.eye(4))
        nib.save(nii_kpgt, save_dir + '/gt2kp.nii.gz')

        '''

        # print('target : ', target.shape)
        # print('ground_truth : ', ground_truth.shape)
        loss_heatmap1 = criterion2(target1, ground_truth1)
        loss_heatmap2 = criterion2(target2, ground_truth2)
        loss_heatmap3 = criterion2(target3, ground_truth3)

        # pred_points = utils.heatmap2kp(target)
        # pred_points = pred_points.cuda()
        # pred_points = pred_points / 128

        # gt_points = utils.heatmap2kp(ground_truth)
        # gt_points = gt_points.cuda()
        # gt_points = gt_points / 128

        # loss_pts = criterion3(pred_points, gt_points)

        # losses = (loss_pts / alpha_pts) + (alpha_heatmap * loss_heatmap)
        losses = loss_heatmap1 + loss_heatmap2 + loss_heatmap3
        # print(' {} / {} => Point loss : {} | Heatmap loss : {} | Total loss : {}'.format(idx+1, len(train_dataloader), loss_pts.item() / alpha_pts, loss_heatmap.item() * alpha_heatmap, losses.item()))
        print(' {} / {} => Heatmap1 loss : {} | Heatmap2 loss : {} | Heatmap3 loss : {} | Total loss : {}'.format(
            idx+1, len(train_dataloader), loss_heatmap1.item(), loss_heatmap2.item(), loss_heatmap3.item(), losses.item()
        ))
        # print(' {} / {} => Total loss : {}'.format(idx+1, len(train_dataloader), losses.item()))

        losses.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_loss += losses.item()
    
    scheduler.step()

    valid_losses = 0.0
    model.eval()
    with torch.no_grad():
        for val_idx, data in enumerate(val_dataloader):
            image, ground_truth = data['image'], data['label']
            
            # target = model(image)
            target1, target2, target3 = model(image)

            target = target3

            ground_truth1 = utils.resize_gt(ground_truth, (1, 4, 16, 32, 32))
            ground_truth2 = utils.resize_gt(ground_truth, (1, 4, 32, 64, 64))
            ground_truth3 = ground_truth

            if val_idx < 2:
                save_result(image, target, ground_truth, data['label_array'], val_idx)

            
            loss_heatmap1 = criterion2(target1, ground_truth1)
            loss_heatmap2 = criterion2(target2, ground_truth2)
            loss_heatmap3 = criterion2(target3, ground_truth3)

            # pred_points = utils.heatmap2kp(target)
            # pred_points = pred_points.cuda()
            # pred_points = pred_points / 128

            # gt_points = utils.heatmap2kp(ground_truth)
            # gt_points = gt_points.cuda()
            # gt_points = gt_points / 128

            # loss_pts = criterion3(pred_points, gt_points)

            # valid_loss = (loss_pts / alpha_pts) + (alpha_heatmap * loss_heatmap)
            # print(' {} / {} => Point loss : {} | Heatmap loss : {} | Total loss : {}'.format(val_idx+1, len(val_dataloader), loss_pts.item(), loss_heatmap.item(), valid_loss.item()))
            # valid_loss = loss_heatmap
            valid_loss = loss_heatmap1 + loss_heatmap2 + loss_heatmap3

            # valid_loss = 10 * valid_loss

            # valid_loss = (loss_pts / 10) + (loss_heatmap * 10)
            # valid_loss = loss_heatmap

            # print('Val Point Loss : {}, Heatmap Loss : {}'.format((loss_pts / 10), (loss_heatmap * 10)))
            valid_losses += valid_loss

            
        writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
        writer.add_scalar("Loss/Validation", valid_losses / len(val_dataloader), epoch)
        
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_losses / len(val_dataloader)}')
        
        valid_losses = valid_losses / len(val_dataloader)

        if min_valid_loss > valid_losses:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_losses:.6f}) \t Saving The Model')
            min_valid_loss = valid_losses
            # Saving State Dict
            torch.save(model.state_dict(), f'checkpoints/checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')

writer.flush()
writer.close()