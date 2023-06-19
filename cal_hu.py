import os
import json
import nibabel as nib
import numpy as np

'''
image_arr :  (170, 371, 370)
label_arr :  (170, 371, 370)

MAX, MEAN, MIN
5264.0
-46.725989475449964
-1000.0

1.0
0.0010297010186022394
0.0


MAX :  3476.0
MIN :  -718.9390058606468

'''

DATA_PATH = os.path.abspath('/media/jeeheon/SSD/ToothFairy_Dataset/ban_nii')
data_types = ['train', 'test', 'val']

MAX = -1
MIN = 1e9

voxel_lists = []

for data_type in data_types:
    data_dir = os.path.join(DATA_PATH, data_type)
    file_names = os.listdir(data_dir)

    for idx, file_name in enumerate(file_names):
        cur_each_dir = os.path.join(data_dir, file_name)
        print(idx, ' => ', cur_each_dir)

        image_file = nib.load(cur_each_dir + '/data.nii.gz')
        image_arr = image_file.get_fdata()

        nifti_file = nib.load(cur_each_dir + '/gt_alpha.nii.gz')
        label_arr = nifti_file.get_fdata()

        pos_mask = (label_arr == 1)
        pos_mask = pos_mask * image_arr

        XX, YY, ZZ = image_arr.shape

        for x in range(XX):
            for y in range(YY):
                for z in range(ZZ):
                    if pos_mask[x][y][z] != 0.0:
                        voxel_lists.append(int(pos_mask[x][y][z]))

        print(voxel_lists)
        print('len : ', len(voxel_lists))

        break
    break

bins = np.arange(min(voxel_lists), max(voxel_lists), 1)
hist2, bin_edges = np.histogram(voxel_lists, bins)

plt.hist(weight2, bins2, rwidth = 0.8, color = 'red', alpha = 0.5)
plt.grid()
plt.xlabel('Weight (kg)', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

#         # max_value = np.max(pos_mask)
#         # min_value = np.min(pos_mask)

#         # if MAX < max_value:
#         #     MAX = max_value
#         # if MIN > min_value:
#         #     MIN = min_value

# print('MAX : ', MAX)
# print('MIN : ', MIN)