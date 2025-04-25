import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from sgm.data.my_dataset import *
from gs_util.bounding import *
from torchvision.transforms import ToTensor


def sample(
    input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
    num_frames: Optional[int] = None,  # 21 for SV3D
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    elevations_deg: Optional[float | List[float]] = 10.0,  # For SV3D
    azimuths_deg: Optional[List[float]] = None,  # For SV3D
    image_frame_ratio: Optional[float] = None,
    verbose: Optional[bool] = False,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/demo/")
        model_config = "scripts/sampling/configs/demo.yaml"
        tgt_size = [512, 384]

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    torch.manual_seed(seed)

    gs_out_root = 'gs/output'
    gs_bds_root = 'gs/bds/add'
    scenes = ['kitchen', 'garden']
    for scene in scenes:

        bd_box = torchMesh(os.path.join(gs_bds_root, f'{scene}.obj'))
        cam_center = bd_box.center.detach().cpu().numpy()
        dir1_root = os.path.join(gs_out_root, scene, 'bds', 'ours_30000')
        dir2_root = os.path.join(gs_out_root, scene, 'bds_reverse', 'ours_30000')
        output_scene = os.path.join(output_folder, scene)

        use_depth = os.path.exists(os.path.join(dir1_root, 'depth'))
        cond_frame = load_img(f'{dir1_root}/sd.png', target_size=tgt_size)

        def get_gs_batch(scene_root):

            batch = {}

            frames, masks = [], []
            if use_depth: controls = []
            for i in range(num_frames):
                v_id = '%02d' % (i)
                frames.append(load_img(f'{scene_root}/renders/{v_id}.png', target_size=tgt_size))
                if use_depth: controls.append(load_img(f'{scene_root}/depth/{v_id}.png', target_size=tgt_size, scale=False))
                masks.append(load_img(f'{scene_root}/mask/{v_id}.png', target_size=tgt_size, to_rgb=False, scale=False))
            poses = np.load(f'{scene_root}/poses.npy')

            frames = torch.stack(frames)
            if use_depth: controls = torch.stack(controls)
            masks = torch.stack(masks)

            azimuths, polars, radius = compute_poses(poses, cam_center=cam_center)

            first_frame = cond_frame[None, ...].clone()

            bg_masks = 1. - masks
            if use_depth: control_hint = torch.cat([controls, masks, frames * bg_masks, bg_masks], dim=1)
            else: control_hint = frames * bg_masks

            batch['jpg'] = frames
            batch['control_hint'] = control_hint
            batch['masks'] = masks
            batch['cond_frames_without_noise'] = first_frame
            batch['cond_frames'] = first_frame
            batch['polars_rad'] = torch.tensor(polars)
            batch['azimuths_rad'] = torch.tensor(azimuths)
            batch['rad'] = torch.tensor(radius)

            batch['image_only_indicator'] = torch.zeros([num_frames])
            batch['num_video_frames'] = torch.tensor(num_frames)

            for k, v in batch.items():
                batch[k] = batch[k][None, ...].to(torch.float16).to(device)

            return batch

        dir1_batch = get_gs_batch(dir1_root)
        dir1_out_dir = os.path.join(output_scene, 'bds')
        if not os.path.exists(dir1_out_dir): os.makedirs(dir1_out_dir)
        model.test_infer(dir1_batch, 0, dir1_out_dir)

        dir2_batch = get_gs_batch(dir2_root)
        dir2_out_dir = os.path.join(output_scene, 'bds_reverse')
        if not os.path.exists(dir2_out_dir): os.makedirs(dir2_out_dir)
        model.test_infer(dir2_batch, 0, dir2_out_dir)




def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


if __name__ == "__main__":
    Fire(sample)
