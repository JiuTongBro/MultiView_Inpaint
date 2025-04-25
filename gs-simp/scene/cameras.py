#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.graphics_utils import *

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 mask=None, inpainted=False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)

        if mask is not None: self.mask = mask.clamp(0.0, 1.0).to(self.data_device)
        self.inpainted = inpainted

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.camera_to_world = self.world_view_transform.inverse().transpose(0, 1)

    def update_attr(self, image_name, camera_to_world, original_image=None, mask=None,
                    image_width=384, image_height=512, change_size=True, inpainted=True):

        if change_size:
            focal_x = fov2focal(self.FoVx, self.image_width)
            focal_y = fov2focal(self.FoVy, self.image_height)

            self.FoVx = focal2fov(focal_x, image_width)
            self.FoVy = focal2fov(focal_y, image_height)

        self.image_name = image_name
        self.camera_to_world = camera_to_world

        self.R = self.camera_to_world[:3, :3]
        w2c = torch.linalg.inv(self.camera_to_world)
        self.T = w2c[:3, -1]

        self.world_view_transform = self.camera_to_world.transpose(0, 1).inverse()
        if change_size:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar,
                                                         fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if change_size:
            self.image_width = image_width
            self.image_height = image_height

        if original_image is not None: self.original_image = original_image.clamp(0.0, 1.0).to(self.data_device)
        if mask is not None: self.mask = mask.clamp(0.0, 1.0).to(self.data_device)
        self.inpainted = inpainted

    def update_img_mask(self, image, mask):
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.mask = mask.clamp(0.0, 1.0).to(self.data_device)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

