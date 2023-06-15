import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
import torch.nn.functional as F
import torch.nn as nn

from losses import DiceLoss


if BACKGROUND_AS_CLASS: NUM_CLASSES += 1


writer = SummaryWriter("runs")

model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES)
train_transforms = train_transform
val_transforms = val_transform

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    # model = model.to('cuda')
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda 
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')


train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

# criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS)).cuda()

criterion = DiceLoss()
criterion2 = nn.MSELoss()

optimizer = Adam(params=model.parameters(), lr=0.0005)

min_valid_loss = math.inf

for epoch in range(TRAINING_EPOCH):
    
    train_loss = 0.0
    model.train()

    print('EPOCH : {} / {} , len(train_dataloader) : '.format(epoch, TRAINING_EPOCH), len(train_dataloader))
    for data in train_dataloader:
        image, ground_truth = data['image'], data['label']
        optimizer.zero_grad()

        target = model(image)

        losses = 0.0
        loss = criterion(target, ground_truth)
        bce_loss = criterion2(target, ground_truth)

        # print(loss)
        # print(bce_loss)
        # losses.update({'Dice Loss' : loss.item()})
        # losses.update({'MSE Loss' : bce_loss.item()})
        # losses.update(bce_loss)

        losses = loss + bce_loss

        losses.backward()
        optimizer.step()

        train_loss += losses.item()
    
    valid_loss = 0.0
    model.eval()
    for data in val_dataloader:
        image, ground_truth = data['image'], data['label']
        
        losses = 0.0

        target = model(image)

        loss = criterion(target,ground_truth)
        bce_loss = criterion2(target, ground_truth)

        losses = loss + bce_loss


        valid_loss += losses.item()
        
        
    writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
    writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)
    
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
    
    valid_loss = valid_loss / len(val_dataloader)

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')

writer.flush()
writer.close()

