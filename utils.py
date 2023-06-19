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
    '''
    Need To Improve with Torch.Tensor
    '''

    heatmaps = heatmaps.squeeze(0)

    coords = []
    for heatmap in heatmaps:
        index_flat = np.argmax(heatmap)
        # print(';index_flat : ', index_flat)
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
    grid = grid.unsqueeze(0).cuda() # (1, 64, 128, 128, 3)

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

def voi_crop(x, slice_d, slice_h, slice_w, MARGIN = 10):
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

def hadamard_product_cpu(heatmaps, target2kp, gt_points):
    heatmaps = heatmaps.unsqueeze(4)
    heatmaps = heatmaps.detach().cpu().numpy()

    results = []

    for i in range(4):
        gtkp = gt_points[i]
        tkp = target2kp[i]

        single_heatmap = heatmaps[i]
        size = single_heatmap.shape
        # print('size : ', size)

        d = torch.linspace(0, size[0]-1, size[0])
        h = torch.linspace(0, size[1]-1, size[1])
        w = torch.linspace(0, size[2]-1, size[2])
        
        meshz, meshy, meshx = torch.meshgrid((d,h,w))
        grid = torch.stack((meshz, meshy, meshx), 3)
        grid = grid.numpy()
        # print('grid : ', grid.shape)

        np_sum = np.sum(single_heatmap)
        repeat_single_heatmap = np.repeat(single_heatmap, 3, axis=3)

        # print('repeat_single_heatmap : ', repeat_single_heatmap.shape)

        res = repeat_single_heatmap * grid

        d_sum = np.sum(res[:,:,:,0])
        h_sum = np.sum(res[:,:,:,1])
        w_sum = np.sum(res[:,:,:,2])

        res = [int(d_sum/np_sum), int(h_sum/np_sum), int(w_sum/np_sum)]
        # print('pred : ', tkp, ' => res : ', res)
        # print('gtkp : ', gtkp)
        # print()
        results.append(res)

    return results

def hadamard_product(heatmaps):

    '''
    heatmaps : [1, 4, 64, 128, 128]
    '''

    results = []
    heatmaps = heatmaps.unsqueeze(-1)

    for i in range(4):
        single_heatmap = heatmaps[0, i]
        size = single_heatmap.shape

        d = torch.linspace(0, size[0]-1, size[0])
        h = torch.linspace(0, size[1]-1, size[1])
        w = torch.linspace(0, size[2]-1, size[2])
        
        meshz, meshy, meshx = torch.meshgrid((d,h,w))
        grid = torch.stack((meshz, meshy, meshx), 3).cuda()

        sum = torch.sum(single_heatmap)
        repeat_single_heatmap = single_heatmap.repeat(1, 1, 1, 3)

        res = repeat_single_heatmap * grid
        d_sum = torch.sum(res[:,:,:,0])
        h_sum = torch.sum(res[:,:,:,1])
        w_sum = torch.sum(res[:,:,:,2])

        # results.append([(d_sum/sum), (h_sum/sum), (w_sum/sum)])

        pred_keypoints = torch.stack([(d_sum/sum), (h_sum/sum), (w_sum/sum)], dim=0)
        results.append(pred_keypoints)

    results = torch.stack(results, dim=0)
    # print('results : ', results[0][0], results[0][0].requires_grad)

    return results