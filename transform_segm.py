from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    Spacingd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    AddChanneld,
    DivisiblePadd
)

#Cuda version of "train_transform"
train_transform_cuda = Compose(
    [   
        AddChanneld(keys=["image", "label"]),
        # Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        # RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        # RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        # RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        # DivisiblePadd(k=16, keys=["image", "label"]),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
)

#Cuda version of "val_transform"
val_transform_cuda = Compose(
    [   
        # AddChanneld(keys=["image"]),
        AddChanneld(keys=["image", "label"]),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        # DivisiblePadd(k=16, keys=["image", "label"]),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
)