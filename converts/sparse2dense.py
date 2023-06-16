import os
import json
import nibabel as nib
import numpy as np

DATA_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset/nii_sparse')
SAVE_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset/ban_nii')

# data_types = ['train', 'test', 'val']
data_types = ['val']

for data_type in data_types:
    data_dir = os.path.join(DATA_PATH, data_type)
    save_dir = os.path.join(SAVE_PATH, data_type)

    file_names = os.listdir(save_dir)

    for idx, file_name in enumerate(file_names):
        print(idx, ' => ', file_name)

        cur_each_dir = os.path.join(data_dir, file_name)
        save_each_dir = os.path.join(save_dir, file_name)

        # np_gt = np.load(cur_each_dir + '/gt_alpha.npy')

        nifti_file = nib.load(cur_each_dir + '/gt_sparse.nii.gz')
        # nifti_array = nifti_file.get_fdata()

        # nii_data = nib.Nifti1Image(np_data, affine=np.eye(4))
        # nii_gt = nib.Nifti1Image(np_gt, affine=np.eye(4))

        nib.save(nifti_file, save_each_dir + '/gt_sparse.nii.gz')
        # nib.save(nii_gt, save_each_dir + '/gt_alpha.nii.gz')