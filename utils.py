import numpy as np
import torch
import nibabel as nib

from config import (
    RESIZE_DEPTH, RESIZE_HEIGHT, RESIZE_WIDTH
)

device = torch.device("cuda:0")

def compute_3D_coordinate(target_numpy):
    '''
    입력 영상에 대해, X축 기준으로 절반 나누어 각 Sparse 한 신경에 대한 Start point / End point 연산.
    
    Input : Numpy Array
        Len (numpy array) = 3
        Sequence = Depth / Height / Width = d, y, x

    Output : List
        Len = 4
        Sequence = Coords of D, Y, X
    '''

    mid_x = target_numpy.shape[2] // 2

    left_target = target_numpy[:,:,:mid_x]
    right_target = target_numpy[:, :,mid_x:]

    '''
    lt = nib.Nifti1Image(left_target, affine=np.eye(4))
    nib.save(lt, './left_target.nii.gz')

    rt = nib.Nifti1Image(right_target, affine=np.eye(4))
    nib.save(rt, './right_target.nii.gz')
    '''

    left_target = np.argwhere(left_target > 0)
    right_target = np.argwhere(right_target > 0)

    left_min_coords = left_target[np.argmin(left_target[:, 1])]
    left_max_coords = left_target[np.argmax(left_target[:, 1])]
    right_min_coords = right_target[np.argmin(right_target[:, 1])]
    right_max_coords = right_target[np.argmax(right_target[:, 1])]

    left_min_coords = [left_min_coords[0], left_min_coords[1], left_min_coords[2]]
    left_max_coords = [left_max_coords[0], left_max_coords[1], left_max_coords[2]]

    right_max_coords = [right_max_coords[0], right_max_coords[1], right_max_coords[2]+ mid_x]
    right_min_coords = [right_min_coords[0], right_min_coords[1], right_min_coords[2]+ mid_x]

    target_pts = []
    target_pts.append(left_min_coords)
    target_pts.append(left_max_coords)
    target_pts.append(right_min_coords)
    target_pts.append(right_max_coords)

    return target_pts

def heatmap2kp(heatmaps):
    coords = []
    for heatmap in heatmaps:
        index_flat = np.argmax(heatmap)
        index_3d = np.unravel_index(index_flat, heatmap.shape)
        index_3d = torch.tensor(index_3d, dtype=torch.float32)
        coords.append(index_3d)
        # coords.append([index_3d[0], index_3d[1], index_3d[2]])
        # print('index_3d : ', index_3d)

    coords = torch.stack(coords, dim=0)
    # coords = torch.tensor(coords, dtype=torch.float32)
    return coords


def generate_gaussian_heatmap(size, coord, sigma=2.0):
    d = np.arange(size[0])
    w = np.arange(size[1])
    h = np.arange(size[2])
    
    # wx, hx, dx = np.meshgrid(w, h, d)
    dx, wx, hx = np.meshgrid(d, w, h)
    # p = np.stack((wx, hx, dx), axis=-1)
    # print('p  : ', p)
    # p = np.stack((dx, wx, hx), axis=-1)
    
    # print('sigma sigma : ', sigma)
    # heatmap = np.exp(-np.linalg.norm(p-coord, ord=2, axis=-1) / (sigma**2))
    # heatmap = np.exp(-np.linalg.norm(p-coord, ord=2, axis=-1)) / (2*sigma**2)
    heatmap = np.exp(-((dx-coord[0])**2 + (wx-coord[1])**2 + (hx-coord[2])**2) / (2*sigma**2))

    heatmap = np.transpose(heatmap, (1, 0, 2))
    # heatmap = np.transpose(heatmap, (0, 2, 1))
    # print('heatmap : ', heatmap.shape)
    heatmap = torch.tensor(heatmap).cuda()
    return heatmap


def kp2heatmap(coords, size):
    res = []
    # cnt = 0
    # save_dir='./results_heatmap_mse'

    for coord in coords:
        # heatmap = np.zeros(size)
        heatmap = generate_gaussian_heatmap(size, coord)
        res.append(heatmap)

        # np_image = heatmap.numpy()
        # nii_image = nib.Nifti1Image(np_image, affine=np.eye(4))
        # nib.save(nii_image, save_dir + '/hhhhh{}.nii.gz'.format(cnt))
        # cnt += 1
    heatmaps = torch.stack(res, dim=0)
    heatmaps = heatmaps.float()

    return heatmaps

def resize_img(img, size):
    d = torch.linspace(-1,1,size[0])
    h = torch.linspace(-1,1,size[1])
    w = torch.linspace(-1,1,size[2])
    
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3)
    grid = grid.unsqueeze(0) # (1, 64, 128, 128, 3)

    img = torch.tensor(img).float()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    img = img.permute(0,1,4,3,2)
    # img = torch.nn.functional.grid_sample(img, grid, mode='bilinear', align_corners=True)
    img = torch.nn.functional.grid_sample(img, grid, mode='nearest', align_corners=True)
    # print('img : ', img.shape)
    img = img.squeeze(0).squeeze(0)
    return img

def resize_gt(img, size):
    d = torch.linspace(-1,1,size[2])
    h = torch.linspace(-1,1,size[3])
    w = torch.linspace(-1,1,size[4])
    
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3)
    grid = grid.unsqueeze(0) # (1, 64, 128, 128, 3)

    img = img.permute(0,1,4,3,2)
    # img = torch.nn.functional.grid_sample(img, grid, mode='bilinear', align_corners=True)
    img = torch.nn.functional.grid_sample(img, grid, mode='nearest', align_corners=True)
    # print('img : ', img.shape)
    # img = img.squeeze(0).squeeze(0)
    return img

def resize_tensor(img, size):
    d = torch.linspace(-1,1,size[0])
    h = torch.linspace(-1,1,size[1])
    w = torch.linspace(-1,1,size[2])
    
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3)
    grid = grid.unsqueeze(0) # (1, 64, 128, 128, 3)
    grid = grid.cuda(0)

    img = img.permute(0,1,4,3,2)
    img = torch.nn.functional.grid_sample(img, grid, mode='nearest', align_corners=True)
    # print('img : ', img.shape)
    return img

def voi_crop(x, slice_d, slice_h, slice_w, MARGIN = 0):
    slice_d = sorted(slice_d)
    slice_h = sorted(slice_h)
    slice_w = sorted(slice_w)

    # print('slice : ', slice_d, slice_h, slice_w)

    ds = slice_d[0]-MARGIN
    de = slice_d[1]+MARGIN

    hs = slice_h[0]-MARGIN
    he = slice_h[1]+MARGIN

    ws = slice_w[0]-MARGIN
    we = slice_w[1]+MARGIN

    if ds < 0:
        ds = 0
    if de > RESIZE_DEPTH:
        de = RESIZE_DEPTH - 1

    if hs < 0:
        hs = 0
    if he > RESIZE_HEIGHT:
        he = RESIZE_HEIGHT - 1

    if ws < 0:
        ws = 0
    if we > RESIZE_WIDTH:
        we = RESIZE_WIDTH - 1

    # print('out : ', ds, de, ', ', hs, he, ', ', ws, we)
    # print()

    x = x.squeeze(0)
    x = x[:, ds:de, hs:he, ws:we]
    x = x.unsqueeze(0)

    return x
    

def voi_crop2(x, slice_d, slice_h, slice_w, MARGIN = 10):
    slice_d = sorted(slice_d)
    slice_h = sorted(slice_h)
    slice_w = sorted(slice_w)

    # ds = slice_d[0]-MARGIN
    # de = slice_d[1]+MARGIN

    # hs = slice_h[0]-MARGIN
    # he = slice_h[1]+MARGIN

    # ws = slice_w[0]-MARGIN
    # we = slice_w[1]+MARGIN

    ds = slice_d[0]
    de = slice_d[1]

    hs = slice_h[0]
    he = slice_h[1]

    ws = slice_w[0]
    we = slice_w[1]

    if de == ds:
        if de + MARGIN > RESIZE_DEPTH:
            ds = ds - MARGIN
        else:
            de = de + MARGIN

    if he == hs:
        if he + MARGIN > RESIZE_HEIGHT:
            hs = hs - MARGIN
        else:
            he = he + MARGIN

    if we == ws:
        if we + MARGIN > RESIZE_WIDTH:
            ws = ws - MARGIN
        else:
            we = we + MARGIN

    if (de - ds) > MARGIN:
        if de + MARGIN//2 <= RESIZE_DEPTH:
            de = de + MARGIN//2
        if ds - MARGIN//2 >= 0:
            ds = ds - MARGIN//2

    if (he - hs) > MARGIN:
        if he + MARGIN//2 <= RESIZE_HEIGHT:
            he = he + MARGIN//2
        if hs - MARGIN//2 >= 0:
            hs = hs - MARGIN//2

    if (we - ws) > MARGIN:
        if we + MARGIN//2 <= RESIZE_WIDTH:
            we = we + MARGIN//2
        if ws - MARGIN//2 >= 0:
            ws = ws - MARGIN//2

    x = x.squeeze(0)
    x = x[:, ds:de, hs:he, ws:we]
    x = x.unsqueeze(0)

    return x

def postprocess(x, res, slice_d, slice_h, slice_w):
    slice_d = sorted(slice_d)
    slice_h = sorted(slice_h)
    slice_w = sorted(slice_w)

    ds = slice_d[0]
    de = slice_d[1]

    hs = slice_h[0]
    he = slice_h[1]

    ws = slice_w[0]
    we = slice_w[1]

    x = resize_tensor(x, (de-ds, he-hs, we-ws))
    x = x.squeeze(0)

    res[:, ds:de, hs:he, ws:we] = x

    return res