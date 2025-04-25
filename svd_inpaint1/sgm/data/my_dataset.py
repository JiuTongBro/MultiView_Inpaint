import os
import glob
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from einops import rearrange, repeat
import pytorch_lightning as pl
from functools import partial
import json

import random
from sgm.util import instantiate_from_config
from gs_util.bounding import *


def compute_poses(poses, cam_center=None):
    cam_poses = poses[:, :3, -1]  # 14, 3
    if cam_center is None:
        cam_center = np.mean(cam_poses, axis=0, keepdims=True)
    cam_dirs = cam_poses - cam_center

    radius = np.linalg.norm(cam_dirs, axis=-1)
    scaled_radius = (radius - radius[0]) / radius[0]

    cam_dirs = cam_dirs / radius[:, None]  # 14, 3
    c2w_r = poses[:, :3, :3]
    c2w_r = c2w_r / np.linalg.norm(c2w_r, axis=-1, keepdims=True)

    sphere_z = - cam_dirs[0]
    sphere_y = np.cross(sphere_z, c2w_r[0, :, 0])
    sphere_x = np.cross(sphere_y, sphere_z)

    polar_error = np.arccos(np.sum(sphere_z * c2w_r[0, :, 2], axis=-1))
    if sphere_z[1] > c2w_r[0, 1, 2]:
        polar_error = - polar_error

    sphere_c2w = np.stack([sphere_x, sphere_y, sphere_z], axis=1)
    sphere_w2c = (sphere_c2w.T)[None, ...]  # 1, 3, 3
    sphere_dirs = sphere_w2c @ (cam_dirs.T)
    sphere_dirs = (sphere_dirs.T)[..., 0]
    sphere_dirs = sphere_dirs / np.linalg.norm(sphere_dirs, axis=-1, keepdims=True)

    azimuths = np.arctan2(sphere_dirs[:, 0], sphere_dirs[:, 2])  # 14, [-pi, pi]
    azimuths = azimuths - azimuths[0]  # 14, [-2*pi, 2*pi]
    azimuths = np.where(azimuths > np.pi, azimuths - 2 * np.pi, azimuths)  # 14, [-2*pi, pi]
    azimuths = np.where(azimuths < -np.pi, azimuths + 2 * np.pi, azimuths)  # 14, [-pi, pi]
    azimuths = np.where(azimuths < -np.pi, azimuths + 2 * np.pi, azimuths)  # 14, [-pi, pi]

    polars = np.arctan(sphere_dirs[:, 1] / np.sqrt(sphere_dirs[:, 0] ** 2 + sphere_dirs[:, 2] ** 2))  # [-pi/2, pi/2]
    polars = polars + polar_error  # 14, [-pi, pi]

    return azimuths, polars, scaled_radius


def compute_poses2(poses, cam_center=None):
    cam_poses = poses[:, :3, -1]  # 14, 3
    if cam_center is None:
        cam_center = np.mean(cam_poses, axis=0, keepdims=True)
    cam_dirs = cam_poses - cam_center

    radius = np.linalg.norm(cam_dirs, axis=-1)
    scaled_radius = (radius - radius[0]) / radius[0]

    cam_dirs = cam_dirs / radius[:, None]  # 14, 3
    c2w_r = poses[:, :3, :3]
    c2w_r = c2w_r / np.linalg.norm(c2w_r, axis=-1, keepdims=True)

    sphere_z = - cam_dirs[0]
    sphere_y = np.cross(sphere_z, c2w_r[0, :, 0])
    sphere_x = np.cross(sphere_y, sphere_z)

    polar_error = np.arccos(np.sum(sphere_z * c2w_r[0, :, 2], axis=-1))
    if sphere_z[1] > c2w_r[0, 1, 2]:
        polar_error = - polar_error

    sphere_c2w = np.stack([sphere_x, sphere_y, sphere_z], axis=1)
    sphere_w2c = (sphere_c2w.T)[None, ...]  # 1, 3, 3
    sphere_dirs = sphere_w2c @ (cam_dirs.T)
    sphere_dirs = (sphere_dirs.T)[..., 0]
    sphere_dirs = sphere_dirs / np.linalg.norm(sphere_dirs, axis=-1, keepdims=True)

    azimuths = np.arctan2(sphere_dirs[:, 0], sphere_dirs[:, 2])  # 14, [-pi, pi]
    azimuths = azimuths - azimuths[0]  # 14, [-2*pi, 2*pi]
    azimuths = np.where(azimuths > np.pi, azimuths - 2 * np.pi, azimuths)  # 14, [-2*pi, pi]
    azimuths = np.where(azimuths < -np.pi, azimuths + 2 * np.pi, azimuths)  # 14, [-pi, pi]
    azimuths = np.where(azimuths < -np.pi, azimuths + 2 * np.pi, azimuths)  # 14, [-pi, pi]

    polars = np.arctan(sphere_dirs[:, 1] / np.sqrt(sphere_dirs[:, 0] ** 2 + sphere_dirs[:, 2] ** 2))  # [-pi/2, pi/2]
    # print(np.rad2deg(polar_error))
    polars = (polars + np.pi / 2) - polar_error

    return azimuths % (2 * np.pi), polars % np.pi, scaled_radius


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 num_workers=None, shuffle_test_loader=False, shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader

    def prepare_data(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers)


class SingleVideoDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

    def __len__(self):
        return int(self.repeat)
    

    def __getitem__(self, index):
        batch = {}

        frames = []
        for i in range(self.num_frames):
            frames.append( load_img(f'{self.data_root}/{i}.png', target_size=self.size) )
        frames = torch.stack(frames)

        first_frame = frames[[0]].clone()

        batch['jpg'] = frames
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames) )
        batch['motion_bucket_id'] = ( torch.tensor([self.motion_bucket_id]).repeat(self.num_frames) )
        batch['cond_aug'] = repeat( torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames )
        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class SingleControlledVideoDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

    def __len__(self):
        return int(self.repeat)

    def __getitem__(self, index):
        batch = {}

        frames = []
        for i in range(self.num_frames):
            frames.append( load_img(f'{self.data_root}/img/{i}.png', target_size=self.size) )
        frames = torch.stack(frames)
        controls = []
        for i in range(self.num_frames):
            skt = load_img(f'{self.data_root}/skt/{i}.png', target_size=self.size)
            controls.append( -skt * 0.5 + 0.5 )
        controls = torch.stack(controls)

        first_frame = frames[[0]].clone()

        batch['jpg'] = frames
        batch['control_hint'] = controls
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['fps_id'] = ( torch.tensor([self.fps_id]).repeat(self.num_frames) )
        batch['motion_bucket_id'] = ( torch.tensor([self.motion_bucket_id]).repeat(self.num_frames) )
        batch['cond_aug'] = repeat( torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames )
        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# w/ label_embed finetune

class VideoInpaintDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size) )
            controls.append(load_img(f'{scene_root}/depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        with open(f'{scene_root}/instance.json') as f:
            json_f = json.load(f)
        text = json_f['text']

        ascii_text = [ord(c) for c in text]
        if len(ascii_text) < 64:
            ascii_text += [ord('#') for _ in range(len(ascii_text), 64)]
        text_tensor = torch.tensor(ascii_text)

        first_frame = frames[[0]].clone()

        batch['jpg'] = frames
        batch['control_hint'] = controls
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        # batch['cond_frames_without_noise'] = controls * 2. - 1.
        # batch['cond_frames'] = controls * 2. - 1. + self.cond_aug * torch.randn_like(controls)
        batch['fps_id'] = ( torch.tensor([self.fps_id]).repeat(self.num_frames) )
        # print(batch['fps_id'])
        batch['motion_bucket_id'] = ( torch.tensor([self.motion_bucket_id]).repeat(self.num_frames) )
        batch['cond_aug'] = repeat( torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames )

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class VideoInpaintDataset2(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        poses = np.load(f'{scene_root}/poses.npy')
        azimuths, polars, radius = compute_poses(poses)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['polars_rad'] = (torch.tensor(polars))
        batch['azimuths_rad'] = (torch.tensor(azimuths))
        batch['rad'] = (torch.tensor(radius))
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class SV3DInpaintDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        frames = frames[1:] + frames[:1]
        controls = controls[1:] + controls[:1]
        masks = masks[1:] + masks[:1]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        poses = np.load(f'{scene_root}/poses.npy')
        azimuths, polars, radius = compute_poses2(poses)

        azimuths = np.append(azimuths[1:], azimuths[:1])
        polars = np.append(polars[1:], polars[:1])
        radius = np.append(radius[1:], radius[:1])

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['polars_rad'] = (torch.tensor(polars))
        batch['azimuths_rad'] = (torch.tensor(azimuths))
        batch['rad'] = (torch.tensor(radius))
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class VideoLeastDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, masks = [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        frames = torch.stack(frames)
        masks = torch.stack(masks)

        azimuths, polars, radius = compute_poses(poses, cam_center=cam_center)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = frames * bg_masks

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['polars_rad'] = (torch.tensor(polars))
        batch['azimuths_rad'] = (torch.tensor(azimuths))
        batch['rad'] = (torch.tensor(radius))

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class SV3DLeastDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, masks = [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        # [1..n] -> [n..1]
        frames.reverse()
        masks.reverse()
        poses = poses[::-1, ...]

        frames = frames[1:] + frames[:1]
        masks = masks[1:] + masks[:1]

        frames = torch.stack(frames)
        masks = torch.stack(masks)

        azimuths, polars, radius = compute_poses2(poses, cam_center=cam_center)

        azimuths = np.append(azimuths[1:], azimuths[:1])
        polars = np.append(polars[1:], polars[:1])
        radius = np.append(radius[1:], radius[:1])

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = frames * bg_masks

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['polars_rad'] = (torch.tensor(polars))
        batch['azimuths_rad'] = (torch.tensor(azimuths))
        batch['rad'] = (torch.tensor(radius))

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class VideoForwardDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        azimuths, polars, radius = compute_poses(poses, cam_center=cam_center)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['polars_rad'] = (torch.tensor(polars))
        batch['azimuths_rad'] = (torch.tensor(azimuths))
        batch['rad'] = (torch.tensor(radius))
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class VideoForwardLeastDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        azimuths, polars, radius = compute_poses(poses, cam_center=cam_center)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = frames * bg_masks

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['polars_rad'] = (torch.tensor(polars))
        batch['azimuths_rad'] = (torch.tensor(azimuths))
        batch['rad'] = (torch.tensor(radius))
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class EstVideoForwardDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/est_depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        azimuths, polars, radius = compute_poses(poses, cam_center=cam_center)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['polars_rad'] = (torch.tensor(polars))
        batch['azimuths_rad'] = (torch.tensor(azimuths))
        batch['rad'] = (torch.tensor(radius))
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class EstVideoForwardDataset2(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/est_depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        azimuths, polars, radius = compute_poses(poses, cam_center=cam_center)

        first_frame = frames[[0]].clone()

        control_hint = torch.cat([controls, masks,], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['polars_rad'] = (torch.tensor(polars))
        batch['azimuths_rad'] = (torch.tensor(azimuths))
        batch['rad'] = (torch.tensor(radius))
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class GS_VideoForwardLeastDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 scenes=[]
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode_list = ['x1', 'x2', 'y1', 'y2', 'xy11', 'xy12', 'xy21', 'xy22']

        self.scene_ids = scenes
        print('\n # Dataset Len:', len(self.scene_ids) * len(self.mode_list))

    def __len__(self):
        return len(self.scene_ids) * len(self.mode_list)
        # return int(self.repeat)

    def __getitem__(self, index):
        mode_id = index % len(self.mode_list)
        scene = self.scene_ids[int(index // len(self.mode_list))]

        scene_root = os.path.join(self.data_root, scene, self.mode_list[mode_id], 'ours_30000')
        cond_frame = load_img(os.path.join(self.data_root, scene, 'x1', 'ours_30000', 'sd.png'), target_size=self.size)

        batch = {}

        frames, masks = [], []
        for i in range(self.num_frames):
            v_id = '%02d' % (i)
            frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=self.size))
            masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        frames = torch.stack(frames)
        masks = torch.stack(masks)

        azimuths, polars, radius = compute_poses(poses, cam_center=cam_center)

        first_frame = cond_frame[None, ...].clone()

        bg_masks = 1. - masks
        control_hint = frames * bg_masks

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['polars_rad'] = (torch.tensor(polars))
        batch['azimuths_rad'] = (torch.tensor(azimuths))
        batch['rad'] = (torch.tensor(radius))
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch

# for old inpainting with bds and bds_inverse
class GS_VideoForwardLeastDataset2(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 scenes=[]
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = scenes
        print('\n # Dataset Len:', len(self.scene_ids) * 2)

    def __len__(self):
        return len(self.scene_ids) * 2
        # return int(self.repeat)

    def __getitem__(self, index):
        reverse = (index % 2) != 1
        scene = self.scene_ids[int(index // 2)]

        if reverse:
            scene_root = os.path.join(self.data_root, scene, 'bds_reverse', 'ours_30000')
        else:
            scene_root = os.path.join(self.data_root, scene, 'bds', 'ours_30000')

        cond_frame = load_img(os.path.join(self.data_root, scene, 'bds', 'ours_30000', 'sd.png'), target_size=self.size)

        batch = {}

        frames, masks = [], []
        for i in range(self.num_frames):
            v_id = '%02d' % (i)
            frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=self.size))
            masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        frames = torch.stack(frames)
        masks = torch.stack(masks)

        azimuths, polars, radius = compute_poses(poses, cam_center=cam_center)

        first_frame = cond_frame[None, ...].clone()

        bg_masks = 1. - masks
        control_hint = frames * bg_masks

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['polars_rad'] = (torch.tensor(polars))
        batch['azimuths_rad'] = (torch.tensor(azimuths))
        batch['rad'] = (torch.tensor(radius))
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch

# for ctrl
class GS_VideoForwardLeastDataset3(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 scenes=[]
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode_list = ['x1', 'x2', 'y1', 'y2', 'xy11', 'xy12', 'xy21', 'xy22']

        self.scene_ids = scenes
        print('\n # Dataset Len:', len(self.scene_ids) * len(self.mode_list))

    def __len__(self):
        return len(self.scene_ids) * len(self.mode_list)
        # return int(self.repeat)

    def __getitem__(self, index):
        mode_id = index % len(self.mode_list)
        scene = self.scene_ids[int(index // len(self.mode_list))]

        scene_root = os.path.join(self.data_root, scene, self.mode_list[mode_id], 'ours_30000')
        cond_frame = load_img(os.path.join(self.data_root, scene, 'x1', 'ours_30000', 'sd.png'), target_size=self.size)

        batch = {}

        frames, masks = [], []
        for i in range(self.num_frames):
            v_id = '%02d' % (i)
            frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=self.size))
            masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        frames = torch.stack(frames)
        masks = torch.stack(masks)

        first_frame = cond_frame[None, ...].clone()

        bg_masks = 1. - masks
        control_hint = frames * bg_masks

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class GS_VideoForwardDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 scenes=[]
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = scenes
        print('\n # Dataset Len:', len(self.scene_ids) * 2)

    def __len__(self):
        return len(self.scene_ids) * 2
        # return int(self.repeat)

    def __getitem__(self, index):
        reverse = (index % 2) != 1
        scene = self.scene_ids[int(index // 2)]

        if reverse: scene_root = os.path.join(self.data_root, scene, 'bds_reverse', 'ours_30000')
        else: scene_root = os.path.join(self.data_root, scene, 'bds', 'ours_30000')

        cond_frame = load_img(os.path.join(self.data_root, scene, 'bds', 'ours_30000', 'sd.png'), target_size=self.size)

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%02d' % (i)
            frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        azimuths, polars, radius = compute_poses(poses, cam_center=cam_center)

        first_frame = cond_frame[None, ...].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['polars_rad'] = (torch.tensor(polars))
        batch['azimuths_rad'] = (torch.tensor(azimuths))
        batch['rad'] = (torch.tensor(radius))
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class GS_VideoForwardDataset2(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 scenes=[]
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        # self.mode_list = ['x1', 'x2', 'y1', 'y2', 'xy11', 'xy12', 'xy21', 'xy22']
        self.mode_list = ['x1', 'x2',]

        self.scene_ids = scenes
        print('\n # Dataset Len:', len(self.scene_ids) * len(self.mode_list))

    def __len__(self):
        return len(self.scene_ids) * len(self.mode_list)
        # return int(self.repeat)

    def __getitem__(self, index):
        mode_id = index % len(self.mode_list)
        scene = self.scene_ids[int(index // len(self.mode_list))]

        scene_root = os.path.join(self.data_root, 'seq', scene, self.mode_list[mode_id], 'ours_30000')
        depth_root = os.path.join(self.data_root, 'depth', scene, self.mode_list[mode_id])
        cond_frame = load_img(os.path.join(self.data_root, 'ctrl', scene, 'ctrl.png'), target_size=self.size)

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%02d' % (i)
            frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{depth_root}/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = cond_frame[None, ...].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch

# gt disparity
class GS_VideoForwardDataset3(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 scenes=[]
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        # self.mode_list = ['x1', 'x2', 'y1', 'y2', 'xy11', 'xy12', 'xy21', 'xy22']
        self.mode_list = ['x1', 'x2',]

        self.scene_ids = scenes
        print('\n # Dataset Len:', len(self.scene_ids) * len(self.mode_list))

    def __len__(self):
        return len(self.scene_ids) * len(self.mode_list)
        # return int(self.repeat)

    def __getitem__(self, index):
        mode_id = index % len(self.mode_list)
        scene = self.scene_ids[int(index // len(self.mode_list))]

        scene_root = os.path.join(self.data_root, 'seq', scene, self.mode_list[mode_id], 'ours_5000')
        cond_frame = load_img(os.path.join(self.data_root, 'ctrl', scene, 'ctrl.png'), target_size=self.size)

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%02d' % (i)
            frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/disparity/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = cond_frame[None, ...].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# ctrl and adjust, inpaint
class GS_VideoForwardDataset4(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 scenes=[]
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        # self.mode_list = ['x1', 'x2', 'y1', 'y2', 'xy11', 'xy12', 'xy21', 'xy22']
        self.mode_list = ['x1', 'x2',]

        self.scene_ids = scenes
        print('\n # Dataset Len:', len(self.scene_ids) * len(self.mode_list))

    def __len__(self):
        return len(self.scene_ids) * len(self.mode_list)
        # return int(self.repeat)

    def __getitem__(self, index):
        mode_id = index % len(self.mode_list)
        scene = self.scene_ids[int(index // len(self.mode_list))]

        scene_root = os.path.join(self.data_root, 'seq', scene, self.mode_list[mode_id], 'ours_30000')
        depth_root = os.path.join(self.data_root, 'depth', scene, self.mode_list[mode_id])
        cond_frame = load_img(os.path.join(self.data_root, 'ctrl', scene, 'ctrl.png'), target_size=self.size)

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%02d' % (i)
            frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{depth_root}/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = cond_frame[None, ...].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


class GS_VideoForwardDataset_Demo(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 scenes=[]
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        # self.mode_list = ['x1', 'x2', 'y1', 'y2', 'xy11', 'xy12', 'xy21', 'xy22']
        self.mode_list = ['x1', 'x2',]

        self.scene_ids = scenes
        print('\n # Dataset Len:', len(self.scene_ids) * len(self.mode_list))

    def __len__(self):
        return 1
        # return len(self.scene_ids) * len(self.mode_list)
        # return int(self.repeat)

    def __getitem__(self, index):
        mode_id = index % len(self.mode_list)
        scene = self.scene_ids[int(index // len(self.mode_list))]

        scene_root = os.path.join(self.data_root, 'seq', scene, self.mode_list[mode_id], 'ours_30000')
        depth_root = os.path.join(self.data_root, 'depth', scene, self.mode_list[mode_id])
        cond_frame = load_img(os.path.join(self.data_root, 'ctrl', scene, 'ctrl.png'), target_size=self.size)

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%02d' % (i)
            frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{depth_root}/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = cond_frame[None, ...].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)
        # control_hint = torch.cat([controls, masks, frames, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# simp
class GS_VideoForwardDatasetSimp(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        # self.mode_list = ['x1', 'x2', 'y1', 'y2', 'xy11', 'xy12', 'xy21', 'xy22']
        self.mode_list = ['x1', 'x2',]

        scenes = os.listdir(os.path.join(self.data_root, 'ctrl1'))
        scenes.sort()

        self.scene_ids = []
        for scene in scenes:
            ctrls = os.listdir(os.path.join(self.data_root, 'ctrl1', scene))
            ctrls.sort()
            for ctrl in ctrls:
                self.scene_ids.append([scene, ctrl])

        print('\n # Dataset Len:', len(self.scene_ids) * len(self.mode_list), len(self.mode_list))

    def __len__(self):
        return len(self.scene_ids) * len(self.mode_list)
        # return int(self.repeat)

    def __getitem__(self, index):
        mode_id = index % len(self.mode_list)
        scene, f_ctrl = self.scene_ids[int(index // len(self.mode_list))]

        scene_root = os.path.join(self.data_root, 'seq', scene, self.mode_list[mode_id], 'ours_30000')
        depth_root = os.path.join(self.data_root, 'depth', scene, self.mode_list[mode_id])
        cond_frame = load_img(os.path.join(self.data_root, 'ctrl1', scene, f_ctrl), target_size=self.size)

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%02d' % (i)
            frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{depth_root}/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = cond_frame[None, ...].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch

# simp, No Depth
class GS_VideoForwardDatasetSimpNoDepth(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        # self.mode_list = ['x1', 'x2', 'y1', 'y2', 'xy11', 'xy12', 'xy21', 'xy22']
        self.mode_list = ['x1', 'x2']

        scenes = os.listdir(os.path.join(self.data_root, 'ctrl1'))
        scenes.sort()

        self.scene_ids = []
        for scene in scenes:
            ctrls = os.listdir(os.path.join(self.data_root, 'ctrl1', scene))
            ctrls.sort()
            for ctrl in ctrls:
                self.scene_ids.append([scene, ctrl])

        print('\n # Dataset Len:', len(self.scene_ids) * len(self.mode_list))

    def __len__(self):
        return len(self.scene_ids) * len(self.mode_list)
        # return int(self.repeat)

    def __getitem__(self, index):
        mode_id = index % len(self.mode_list)
        scene, f_ctrl = self.scene_ids[int(index // len(self.mode_list))]

        scene_root = os.path.join(self.data_root, 'seq', scene, self.mode_list[mode_id], 'ours_30000')
        depth_root = os.path.join(self.data_root, 'depth', scene, self.mode_list[mode_id])
        cond_frame = load_img(os.path.join(self.data_root, 'ctrl1', scene, f_ctrl), target_size=self.size)

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%02d' % (i)
            frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{depth_root}/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = cond_frame[None, ...].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([masks, frames * bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch

# simp, No BG
class GS_VideoForwardDatasetSimpNobg(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        # self.mode_list = ['x1', 'x2', 'y1', 'y2', 'xy11', 'xy12', 'xy21', 'xy22']
        self.mode_list = ['x1', 'x2']

        scenes = os.listdir(os.path.join(self.data_root, 'ctrl1'))
        scenes.sort()

        self.scene_ids = []
        for scene in scenes:
            ctrls = os.listdir(os.path.join(self.data_root, 'ctrl1', scene))
            ctrls.sort()
            for ctrl in ctrls:
                self.scene_ids.append([scene, ctrl])

        print('\n # Dataset Len:', len(self.scene_ids) * len(self.mode_list))

    def __len__(self):
        return len(self.scene_ids) * len(self.mode_list)
        # return int(self.repeat)

    def __getitem__(self, index):
        mode_id = index % len(self.mode_list)
        scene, f_ctrl = self.scene_ids[int(index // len(self.mode_list))]

        scene_root = os.path.join(self.data_root, 'seq', scene, self.mode_list[mode_id], 'ours_30000')
        depth_root = os.path.join(self.data_root, 'depth', scene, self.mode_list[mode_id])
        cond_frame = load_img(os.path.join(self.data_root, 'ctrl1', scene, f_ctrl), target_size=self.size)

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%02d' % (i)
            frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{depth_root}/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = cond_frame[None, ...].clone()

        control_hint = torch.cat([controls, masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch

# simp, No Mask
class GS_VideoForwardDatasetSimpNomask(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        # self.mode_list = ['x1', 'x2', 'y1', 'y2', 'xy11', 'xy12', 'xy21', 'xy22']
        self.mode_list = ['x1', 'x2']

        scenes = os.listdir(os.path.join(self.data_root, 'ctrl1'))
        scenes.sort()

        self.scene_ids = []
        for scene in scenes:
            ctrls = os.listdir(os.path.join(self.data_root, 'ctrl1', scene))
            ctrls.sort()
            for ctrl in ctrls:
                self.scene_ids.append([scene, ctrl])

        print('\n # Dataset Len:', len(self.scene_ids) * len(self.mode_list))

    def __len__(self):
        return len(self.scene_ids) * len(self.mode_list)
        # return int(self.repeat)

    def __getitem__(self, index):
        mode_id = index % len(self.mode_list)
        scene, f_ctrl = self.scene_ids[int(index // len(self.mode_list))]

        scene_root = os.path.join(self.data_root, 'seq', scene, self.mode_list[mode_id], 'ours_30000')
        depth_root = os.path.join(self.data_root, 'depth', scene, self.mode_list[mode_id])
        cond_frame = load_img(os.path.join(self.data_root, 'ctrl1', scene, f_ctrl), target_size=self.size)

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%02d' % (i)
            frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{depth_root}/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = cond_frame[None, ...].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, frames * bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# w/o label_embed finetune

class SVDForwardLeastDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = frames * bg_masks

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch

# mask adjust
class SVDForwardLeastDataset2(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode = os.path.basename(self.data_root)

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids), self.mode)

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/depth/{v_id}.png', target_size=self.size, scale=False))

            if self.mode == 'train':
                raw_mask = load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False)
                np_mask = process_mask(raw_mask[0].detach().cpu().numpy(), k_max=0.4)
                masks.append(torch.from_numpy(np_mask).to(raw_mask.device)[None])
            else:
                masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = frames * bg_masks

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# warpping loss
class SVDForwardLeastDataset3(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode = os.path.basename(self.data_root)

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids), self.mode)

    def __len__(self):
        return len(self.scene_ids)
        # return 1
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, depths, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            depths.append(np.array(Image.open(f'{scene_root}/depth/{v_id}.png'), dtype="uint16"))

            if self.mode == 'train':
                raw_mask = load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False)
                np_mask = process_mask(raw_mask[0].detach().cpu().numpy(), k_max=0.4)
                masks.append(torch.from_numpy(np_mask).to(raw_mask.device)[None])
            else:
                masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            depths.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        device = frames.device

        masks = torch.stack(masks)
        poses = torch.from_numpy(poses.copy()).to(device)
        depths = np.array(depths)

        SCALE = 1000.
        depth_max = 5
        depth_min = 0
        k_scale = 8.

        depths = depths / SCALE
        depths = np.clip(depths, depth_min, depth_max)
        depths = torch.from_numpy(depths).to(device)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = frames * bg_masks

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames


        # modified
        w2c_poses = poses.inverse()
        h, w = self.size
        h, w = int(h / k_scale), int(w / k_scale)

        u = torch.linspace(0, w - 1, w).to(device)
        v = torch.linspace(0, h - 1, h).to(device)
        v, u = torch.meshgrid(v, u)

        meta_path = os.path.join(scene_root, 'metadata')
        with open(meta_path, 'r') as f_json:
            meta = json.load(f_json)

        raw_w, raw_h = meta["w"], meta["h"]
        k_resize = self.size[0] / (raw_h * k_scale)
        K = np.array(meta["K"]).reshape(3, 3).T * k_resize
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        depths = depths[:, None, :, :]
        depths = torch.nn.functional.interpolate(depths, (h, w))
        depths_mask = depths[:, 0, :, :] > 0.

        z = depths[:, 0, :, :]  # 14, H, W
        x = (u[None] - cx) * z / fx
        y = (v[None] - cy) * z / fy
        frame_points = torch.stack([x, y, z], dim=-1)  # 14, H, W, 3
        frame_points = frame_points.view(self.num_frames, -1, 3)  # 14, H*W, 3
        frame_points = torch.cat([frame_points, torch.ones_like(frame_points[:, :, :1]).to(device)], dim=-1)  # 14, H*W, 4
        frame_points = poses @ (frame_points.permute(0, 2, 1))  # 14, 4, H*W

        next_points = frame_points[1:]
        prev_poses = w2c_poses[:self.num_frames - 1]

        c_points = prev_poses @ next_points
        c_points = c_points.view(self.num_frames - 1, -1, h, w)
        u = c_points[:, 0] / c_points[:, 2]
        v = c_points[:, 1] / c_points[:, 2]
        u, v = u * fx + cx, v * fy + cy
        frames_uv = torch.stack([u, v], dim=1).floor()

        hit_mask = depths_mask[1:] & (frames_uv[:, 0, :, :] >= 0) & (frames_uv[:, 0, :, :] < w) & (
                    frames_uv[:, 1, :, :] >= 0) & (frames_uv[:, 1, :, :] < h)
        hit_map = torch.where(hit_mask, torch.ones_like(hit_mask), torch.zeros_like(hit_mask))

        hit_mask_uv = hit_mask[:, None, :, :].repeat(1, 2, 1, 1)
        frames_uv = torch.where(hit_mask_uv, frames_uv, torch.zeros_like(frames_uv))
        frames_ind = torch.tensor(frames_uv[:, 1, :, :] * w + frames_uv[:, 0, :, :], dtype=torch.int64)  # N, H, W
        frames_ind = frames_ind[:, None, :].repeat(1, 4, 1, 1)

        batch['hit_map'] = hit_map
        batch['uv_ind'] = frames_ind

        return batch


class EstSVDForwardDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/est_depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# ctrl and adjust
class EstSVDForwardDataset2(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode = os.path.basename(self.data_root)

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/est_depth/{v_id}.png', target_size=self.size, scale=False))

            if self.mode == 'train':
                raw_mask = load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False)
                np_mask = process_mask(raw_mask[0].detach().cpu().numpy(), k_max=0.4)
                masks.append(torch.from_numpy(np_mask).to(raw_mask.device)[None])
            else:
                masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# ctrl and adjust, inpaint
class EstSVDForwardDataset3(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode = os.path.basename(self.data_root)

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, inpainted_frames, controls, masks = [], [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            inpainted_frames.append(load_img(f'{scene_root}/inpainted/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/est_depth/{v_id}.png', target_size=self.size, scale=False))

            if self.mode == 'train':
                raw_mask = load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False)
                np_mask = process_mask(raw_mask[0].detach().cpu().numpy(), k_max=0.4)
                masks.append(torch.from_numpy(np_mask).to(raw_mask.device)[None])
            else:
                masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            inpainted_frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        inpainted_frames = torch.stack(inpainted_frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, inpainted_frames, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# ctrl, single mask
class EstSVDForwardDatasetSimp(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode = os.path.basename(self.data_root)

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/est_depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# ctrl, single mask
class EstSVDForwardDatasetNodepth(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode = os.path.basename(self.data_root)

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/est_depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([masks, frames * bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# ctrl, single mask
class EstSVDForwardDatasetNobg(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode = os.path.basename(self.data_root)

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/est_depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# ctrl, single mask
class EstSVDForwardDatasetNomask(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode = os.path.basename(self.data_root)

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/est_depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, frames * bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# 360
class EstSVDDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode = os.path.basename(self.data_root)

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/est_depth/{v_id}.png', target_size=self.size, scale=False))

            if self.mode == 'train':
                raw_mask = load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False)
                np_mask = process_mask(raw_mask[0].detach().cpu().numpy(), k_max=0.4)
                masks.append(torch.from_numpy(np_mask).to(raw_mask.device)[None])
            else:
                masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# dynamic
class EstSVDDatasetDynamic(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode = os.path.basename(self.data_root)

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/est_depth/{v_id}.png', target_size=self.size, scale=False))

            if self.mode == 'train':
                raw_mask = load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False)
                np_mask = process_mask(raw_mask[0].detach().cpu().numpy(), k_max=0.4)
                masks.append(torch.from_numpy(np_mask).to(raw_mask.device)[None])
            else:
                masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# gt disparity
class SVDForwardDataset3(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.mode = os.path.basename(self.data_root)

        self.scene_ids = os.listdir(self.data_root)
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (index)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/disparity/{v_id}.png', target_size=self.size, scale=False))

            if self.mode == 'train':
                raw_mask = load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False)
                np_mask = process_mask(raw_mask[0].detach().cpu().numpy(), k_max=0.4)
                masks.append(torch.from_numpy(np_mask).to(raw_mask.device)[None])
            else:
                masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))

        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)

        batch['fps_id'] = (torch.tensor([self.fps_id]).repeat(self.num_frames))
        batch['motion_bucket_id'] = (torch.tensor([self.motion_bucket_id]).repeat(self.num_frames))
        batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


# blending demo

class BlendingDataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 motion_bucket_id,
                 fps_id,
                 num_frames,
                 cond_aug,
                 repeat=1,
                 ):
        self.cond_aug = cond_aug
        self.data_root = data_root
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.size = size
        self.repeat = repeat

        self.sample_id = 11
        self.scene_ids = [os.listdir(self.data_root)[self.sample_id]]
        print('\n # Dataset Len:', len(self.scene_ids))

    def __len__(self):
        return len(self.scene_ids)
        # return int(self.repeat)

    def __getitem__(self, index):
        scene_id = '%09d' % (self.sample_id)
        scene_root = f'{self.data_root}/{scene_id}'

        batch = {}

        frames, controls, masks = [], [], []
        for i in range(self.num_frames):
            v_id = '%05d' % (i)
            frames.append(load_img(f'{scene_root}/rgb/{v_id}.png', target_size=self.size))
            controls.append(load_img(f'{scene_root}/depth/{v_id}.png', target_size=self.size, scale=False))
            masks.append(load_img(f'{scene_root}/masks/{v_id}.png', target_size=self.size, to_rgb=False, scale=False))
        poses = np.load(f'{scene_root}/poses.npy')
        cam_center = np.load(f'{scene_root}/cam_center.npy')

        inverse_prob = random.random()
        if inverse_prob > 0.5:
            frames.reverse()
            controls.reverse()
            masks.reverse()
            poses = poses[::-1, ...]

        frames = torch.stack(frames)
        controls = torch.stack(controls)
        masks = torch.stack(masks)

        azimuths, polars, radius = compute_poses(poses, cam_center=cam_center)

        first_frame = frames[[0]].clone()

        bg_masks = 1. - masks
        control_hint = frames * bg_masks

        batch['jpg'] = frames
        batch['control_hint'] = control_hint
        batch['masks'] = masks
        # batch['text'] = text_tensor
        batch['cond_frames_without_noise'] = first_frame
        batch['cond_frames'] = first_frame + self.cond_aug * torch.randn_like(first_frame)
        batch['polars_rad'] = (torch.tensor(polars))
        batch['azimuths_rad'] = (torch.tensor(azimuths))
        batch['rad'] = (torch.tensor(radius))
        # batch['cond_aug'] = repeat(torch.tensor([self.cond_aug]), '1 -> b', b=self.num_frames)

        batch['image_only_indicator'] = torch.zeros([self.num_frames])
        batch['num_video_frames'] = self.num_frames

        return batch


def load_img(path, target_size, to_rgb=True, scale=True):
    """Load an image, resize and output -1..1"""
    image = Image.open(path)
    if to_rgb: image = image.convert("RGB")
    else: image = image.convert("L")
    
    if target_size == None:
        tform = transforms.Compose([
                transforms.ToTensor(),
            ])
    else:
        tform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
            ])
        
    image = tform(image)
    if scale: return 2.* image - 1.
    else: return image


def process_mask(mask, k_max=0.4):

    if len(mask.shape) == 3:
        mask = np.max(mask, axis=-1)

    h, w = mask.shape

    row_ids = np.argwhere(np.max(mask, axis=1) > 0.)
    col_ids = np.argwhere(np.max(mask, axis=0) > 0.)

    if len(row_ids) == 0 or len(col_ids) == 0: return None

    row_st, row_ed = row_ids[0], row_ids[-1] + 1
    col_st, col_ed = col_ids[0], col_ids[-1] + 1

    d_h, d_w = row_ed - row_st, col_ed - col_st

    mask_k1 = random.random() * k_max
    mask_k2 = random.random() * k_max
    mask_k3 = random.random() * k_max
    mask_k4 = random.random() * k_max

    h_k1, h_k2, w_k1, w_k2 = int(d_h * mask_k1), int(d_h * mask_k2), int(d_w * mask_k3), int(d_w * mask_k4)

    row_st = max(0, int(row_st + h_k1))
    row_ed = min(h, int(row_ed - h_k2))
    col_st = max(0, int(col_st + w_k1))
    col_ed = min(w, int(col_ed - w_k2))

    new_mask = np.zeros_like(mask)
    new_mask[row_st:row_ed, col_st:col_ed] = 1.

    return new_mask