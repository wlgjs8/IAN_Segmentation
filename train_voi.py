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

gc.collect()
torch.cuda.empty_cache()


writer = SummaryWriter("runs")

detector = Detector(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES)
train_transforms = train_transform
val_transforms = val_transform

MODEL_WEIGHT_PATH = './checkpoints(voi detection)/epoch8_valLoss0.5606752634048462.pth'
# model = torch.load(MODEL_WEIGHT_PATH)
detector.load_state_dict(torch.load(MODEL_WEIGHT_PATH))

model = UNet3D(in_channels=1 , num_classes=1)

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

optimizer = Adam(params=model.parameters())

min_valid_loss = math.inf

detector.eval()
for epoch in range(TRAINING_EPOCH):
    
    train_loss = 0.0
    model.train()

    print('EPOCH : {} / {} , len(train_dataloader) : '.format(epoch, TRAINING_EPOCH), len(train_dataloader))
    for idx, data in enumerate(train_dataloader):
        image, ground_truth = data['image'], data['label']

        optimizer.zero_grad()

        heatmaps = detector(image)
        heatmaps = heatmaps.squeeze(0).detach().cpu().numpy()

        pred2kp = utils.get_maximum_point(heatmaps)
        pred2kp = pred2kp.numpy()
        pred2kp = pred2kp.astype(int)

        losses = 0.0
        for i in range(2):
            voi_start = pred2kp[2*i]
            voi_end = pred2kp[2*i + 1]
            # print(i, 'th : ', voi_start)
            # print(i, 'th : ', voi_end)
            # print()
            voi = utils.voi_crop(image, [voi_start[0], voi_end[0]], [voi_start[1], voi_end[1]], [voi_start[2], voi_end[2]])
            voi = utils.resize_tensor(voi, (64, 128, 128))
            voi = voi.cuda()
            target = model(voi)
            crop_ground_truth = utils.voi_crop(ground_truth, [voi_start[0], voi_end[0]], [voi_start[1], voi_end[1]], [voi_start[2], voi_end[2]])
            crop_ground_truth = utils.resize_tensor(crop_ground_truth, (64, 128, 128))
            crop_ground_truth = crop_ground_truth.cuda()
            loss = criterion(target, crop_ground_truth)
            losses += loss

        # print('losses : ', losses)
        losses.backward()

        optimizer.step()

        train_loss += losses
    
    valid_losses = 0.0
    model.eval()
    with torch.no_grad():
        for data in val_dataloader:
            image, ground_truth = data['image'], data['label']

            optimizer.zero_grad()

            heatmaps = detector(image)
            heatmaps = heatmaps.squeeze(0).detach().cpu().numpy()

            pred2kp = utils.get_maximum_point(heatmaps)
            pred2kp = pred2kp.numpy()
            pred2kp = pred2kp.astype(int)

            valid_loss = 0.0
            for i in range(2):
                voi_start = pred2kp[2*i]
                voi_end = pred2kp[2*i + 1]
                voi = utils.voi_crop(image, [voi_start[0], voi_end[0]], [voi_start[1], voi_end[1]], [voi_start[2], voi_end[2]])
                voi = utils.resize_tensor(voi, (64, 128, 128))
                voi = voi.cuda()
                target = model(voi)
                crop_ground_truth = utils.voi_crop(ground_truth, [voi_start[0], voi_end[0]], [voi_start[1], voi_end[1]], [voi_start[2], voi_end[2]])
                crop_ground_truth = utils.resize_tensor(crop_ground_truth, (64, 128, 128))
                crop_ground_truth = crop_ground_truth.cuda()
                loss = criterion(target, crop_ground_truth)
                valid_loss += loss

            valid_losses += valid_loss

            
        # writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
        # writer.add_scalar("Loss/Validation", valid_losses / len(val_dataloader), epoch)
        
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_losses / len(val_dataloader)}')
        
        valid_losses = valid_losses / len(val_dataloader)

        if min_valid_loss > valid_losses:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_losses:.6f}) \t Saving The Model')
            min_valid_loss = valid_losses
            # Saving State Dict
            torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')

writer.flush()
writer.close()

