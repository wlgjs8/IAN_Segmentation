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

from losses import DiceLoss
import utils

# from test_heatmap import save_result

import nibabel as nib
import numpy as np

import gc
import torch
from torch.optim.lr_scheduler import MultiStepLR

gc.collect()
torch.cuda.empty_cache()


writer = SummaryWriter("runs")

model = UNet3D(in_channels=1 , num_classes= 4)

model = model.cuda()
train_transforms = train_transform_cuda
val_transforms = val_transform_cuda 


train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

# criterion = DiceLoss()
criterion2 = nn.MSELoss()
criterion3 = nn.L1Loss()

optimizer = Adam(params=model.parameters(), lr=1e-3)
scheduler = MultiStepLR(optimizer, [40, 120, 360], gamma=0.1, last_epoch=-1)
# get_last_lr

# from save_heatmap import save_result

min_valid_loss = math.inf
alpha_pts = 100
alpha_heatmap = 100

for epoch in range(TRAINING_EPOCH):
    
    train_loss = 0.0
    model.train()

    print('optimizer.param_groups : ', scheduler.get_last_lr())
    print('EPOCH : {} / {}, LR : {}, len(train_dataloader) : , '.format(epoch, TRAINING_EPOCH, optimizer.param_groups[0]["lr"]), len(train_dataloader))
    for idx, data in enumerate(train_dataloader):
        # if idx > 0:
        #     break
        image, ground_truth = data['image'], data['label']
        target = model(image)

        loss_heatmap = criterion2(target, ground_truth)
        
        gt_points = utils.compute_3D_coordinate(image.squeeze(0).squeeze(0).detach().cpu().numpy())
        pred_points = utils.get_maximum_point(target.squeeze(0).detach().cpu().numpy())

        tensor_gt_points = torch.tensor(gt_points).cuda().float()
        pred_points = pred_points.cuda()

        loss_pts = criterion3(pred_points, tensor_gt_points)

        losses = (loss_pts / alpha_pts) + (alpha_heatmap * loss_heatmap)
        print(' {} / {} => Point loss : {} | Heatmap loss : {} | Total loss : {}'.format(idx+1, len(train_dataloader), loss_pts.item() / alpha_pts, loss_heatmap.item() * alpha_heatmap, losses.item()))

        losses.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_loss += losses.item()
    
    scheduler.step()

    valid_losses = 0.0
    model.eval()
    with torch.no_grad():
        for data in val_dataloader:
            image, ground_truth = data['image'], data['label']
            
            target = model(image)

            loss_heatmap = criterion2(target, ground_truth)
        
            gt_points = utils.compute_3D_coordinate(image.squeeze(0).squeeze(0).detach().cpu().numpy())
            pred_points = utils.get_maximum_point(target.squeeze(0).detach().cpu().numpy())

            tensor_gt_points = torch.tensor(gt_points).cuda().float()
            pred_points = pred_points.cuda()

            loss_pts = criterion3(pred_points, tensor_gt_points)

            valid_loss = (loss_pts / alpha_pts) + (alpha_heatmap * loss_heatmap)
            # print(' {} / {} => Point loss : {} | Heatmap loss : {} | Total loss : {}'.format(idx+1, len(train_dataloader), loss_pts.item() / 10, loss_heatmap.item() * 10, losses.item()))


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
            torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')

writer.flush()
writer.close()

