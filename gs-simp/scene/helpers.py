import numpy as np
import torch
from packaging import version as pver
from utils.graphics_utils import *

img2mse = lambda x, y: torch.mean((x - y) ** 2)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

front_dict = {
              # mip
              'bicycle': '_DSC8756',
              'garden': 'DSC07956',
              'bonsai': 'DSCF5565',
              'kitchen': 'DSCF0657',
              'stump': '_DSC9214',
              'room': 'DSCF4680',
              'counter': 'DSCF5898',

              # spin
              '1': '20220819_104243',
              '2': '20220819_104648',
              '3': '20220819_105148',
              '4': '20220819_105637',
              '7': '20220819_111557',
              '9': '20220819_112827',
              '10': '20220823_095100',
              '12': '20220823_093735(0)',
              'book': '20220811_112812',
              'trash': '20220811_093603',
              }

text_dict = {# Mip-NeRF
             'bicycle_bear': 'a toy bear sitting on the bench',
             'bicycle_dog': 'a toy dog sitting on the bench',
             'kitchen_cup': 'a paper cup on the table',
             'stump_flower': 'a yellow flower',
             'garden_cake': 'a birthday cake on the table',
             'garden_gnome': 'a garden gnome on the table',
             'counter_bread': 'a bread on the table',
             'counter_grinder': 'a pepper grinder on the table',

             # Spin-NeRF
             '2_suitcase': 'a suitcase on the floor',
             '9_trash bin': 'a trash bin on the floor',
             '10_candlestick': 'a candlestick on the bench',
             'trash_school bag': 'a school bag on the floor',
             }


cam_dict = {
            # mip
            'bicycle': {'k_lift': np.pi / 6., 'r_scale': 0.7, 'k_bias': 0., 'view_range': np.pi / 3.},
            'bonsai': {'k_lift': np.pi / 6., 'r_scale': 0.6, 'k_bias': 0., 'view_range': np.pi / 3.},
            'kitchen': {'k_lift': np.pi / 4., 'r_scale': 0.8, 'k_bias': 0., 'view_range': np.pi / 3.},
            'garden': {'k_lift': np.pi / 6.,  'r_scale': 0.7, 'k_bias': 0., 'view_range': np.pi / 3.},
            'stump': {'k_lift': np.pi / 6., 'r_scale': 0.5, 'k_bias': 0., 'view_range': np.pi / 3.},
            'counter': {'k_lift': np.pi / 3., 'r_scale': 0.7, 'k_bias': 0., 'view_range': np.pi / 3.},

            # spin
            '1': {'k_lift': np.pi * 5. / 12., 'r_scale': 0.7, 'k_bias': 0., 'view_range': np.pi / 3.},
            '2': {'k_lift': np.pi * 5. / 12., 'r_scale': 0.7, 'k_bias': 0., 'view_range': np.pi / 24.},
            '3': {'k_lift': np.pi / 6., 'r_scale': 1., 'k_bias': 0., 'view_range': np.pi / 3.},
            '4': {'k_lift': np.pi / 6., 'r_scale': 1., 'k_bias': 0., 'view_range': np.pi / 3.},
            '7': {'k_lift': - np.pi * 11. / 6., 'r_scale': 1.2, 'k_bias': 0., 'view_range': np.pi / 12.},
            '9': {'k_lift': np.pi * 5. / 12., 'r_scale': 0.75, 'k_bias': 0., 'view_range': np.pi / 24.},
            '10': {'k_lift': np.pi / 9., 'r_scale': 0.85, 'k_bias': np.pi / 12., 'view_range': np.pi / 4.},
            '12': {'k_lift': np.pi / 3., 'r_scale': 0.85, 'k_bias': 0., 'view_range': np.pi / 3.},
            'book': {'k_lift': np.pi / 3., 'r_scale': 0.85, 'k_bias': 0., 'view_range': np.pi / 12.},
            'trash': {'k_lift': np.pi / 3., 'r_scale': 0.8, 'k_bias': np.pi / 12., 'view_range': np.pi / 4.},
            }


vis_dict = {
            # mip
            'bicycle': {'k_lift': np.pi / 6., 'r_scale': 0.7, 'k_bias': 0., 'view_range': np.pi / 3.},
            # 'bonsai': {'k_lift': np.pi / 6., 'r_scale': 1., 'k_bias': 0., 'view_range': np.pi / 3.},
            'kitchen': {'k_lift': np.pi / 4., 'r_scale': 0.8, 'k_bias': 0., 'view_range': np.pi / 3.},
            'garden': {'k_lift': np.pi / 6.,  'r_scale': 0.75, 'k_bias': 0., 'view_range': np.pi / 3.},
            'stump': {'k_lift': np.pi / 12., 'r_scale': 0.6, 'k_bias': 0., 'view_range': np.pi / 3.},
            # 'room': {'k_lift': np.pi / 3., 'r_scale': 1., 'k_bias': 0., 'view_range': np.pi / 3.},
            'counter': {'k_lift': np.pi / 3., 'r_scale': 0.7, 'k_bias': 0., 'view_range': np.pi / 3.},

            # spin
            # '1': {'k_lift': np.pi * 5. / 12., 'r_scale': 0.7, 'k_bias': 0., 'view_range': np.pi / 6.},
            '2': {'k_lift': np.pi * 5. / 12., 'r_scale': 0.7, 'k_bias': 0., 'view_range': np.pi / 18.},
            # '3': {'k_lift': np.pi / 6., 'r_scale': 1., 'k_bias': 0., 'view_range': np.pi / 6.},
            # '4': {'k_lift': np.pi / 6., 'r_scale': 1., 'k_bias': 0., 'view_range': np.pi / 6.},
            # '7': {'k_lift': - np.pi * 11. / 6., 'r_scale': 1.2, 'k_bias': 0., 'view_range': np.pi / 12.},
            '9': {'k_lift': np.pi * 5. / 12., 'r_scale': 0.75, 'k_bias': 0., 'view_range': np.pi / 18.},
            '10': {'k_lift': np.pi / 9., 'r_scale': 0.7, 'k_bias': np.pi / 12., 'view_range': np.pi / 18.},
            # '12': {'k_lift': np.pi / 3., 'r_scale': 0.85, 'k_bias': 0., 'view_range': np.pi / 6.},
            'book': {'k_lift': np.pi / 3., 'r_scale': 0.85, 'k_bias': 0., 'view_range': np.pi / 12.},
            'trash': {'k_lift': np.pi / 3., 'r_scale': 0.7, 'k_bias': np.pi / 12., 'view_range': np.pi / 18.},
            }

# bds_dict = {'bicycle': [[0.42, 0.41, -0.11], [1.12, 1.11, 0.59]],}


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def get_rays(view):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = view.data_device
    B, H, W = 1, int(view.image_height), int(view.image_width)
    cx, cy = W // 2, H // 2
    fx = fov2focal(view.FoVx, view.image_width)
    fy = fov2focal(view.FoVy, view.image_height)
    poses = view.camera_to_world[None, ...]

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)  # (B, N, 3)
    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    return rays_o[0], rays_d[0]


def intersect_cube(bounds, ray_o, ray_d):
    """bounds[2,3],ray_o[N,3],ray_d[N,3]"""
    """calculate intersections with 3d bounding box"""
    # norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    norm_d = torch.norm(ray_d, dim=-1, keepdim=True)
    tmin = (bounds[:1] - ray_o) / ray_d  # N,3
    tmax = (bounds[1:2] - ray_o) / ray_d  # N,3

    t1 = torch.minimum(tmin, tmax)  # N,3
    t2 = torch.maximum(tmin, tmax)  # N,3
    near = torch.max(t1, axis=-1)[0] # N
    far = torch.min(t2, axis=-1)[0] # N
    mask_at_box = torch.where(far > near, torch.ones_like(near), torch.zeros_like(near))
    return mask_at_box


def normalize_0_to_1(x):
    max_ = torch.max(x)
    min_ = torch.min(x)
    return (x - min_) / (max_ - min_)

def normalization(x, dim=-1):
    r = torch.norm(x, p=2, dim=dim, keepdim=True)
    return x / r