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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.helpers import *
from utils.bounding import *

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, bd_box, view_id=True):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask")
    masked_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masked")

    makedirs(render_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)
    makedirs(masked_path, exist_ok=True)

    poses = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        if view_id: v_id = view.image_name
        else: v_id = "{0:02d}".format(idx)

        poses.append(view.camera_to_world.detach().cpu().numpy())

        outputs = render(view, gaussians, pipeline, background)
        rendering, depth = outputs["render"], outputs["depth"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{v_id}.png"))

        rays_o, rays_d = get_rays(view)
        _, inter_t, _, _ = bd_box.intersect(rays_o, rays_d)  # n, 1
        inter_t = inter_t.view(int(view.image_height), int(view.image_width))[None, ...]  # 1, h, w
        inter_mask = (inter_t > 0.) & ((inter_t < depth) | (depth == 15.))
        inter_mask = inter_mask.float()
        torchvision.utils.save_image(inter_mask, os.path.join(mask_path, f"{v_id}.png"))

        torchvision.utils.save_image(rendering * (1. - inter_mask) + inter_mask, os.path.join(masked_path, f"{v_id}.png"))

    cam_center = bd_box.center.detach().cpu().numpy()
    np.save(os.path.join(model_path, name, "ours_{}".format(iteration), "cam_center.npy"), cam_center)
    np.save(os.path.join(model_path, name, "ours_{}".format(iteration), "poses.npy"), poses)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams,
                skip_train : bool, skip_test : bool, delete : bool, sds: bool, scene_name: str):

    print(dataset.source_path, dataset.model_path)

    with torch.no_grad():

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, do_delete=delete, sds=sds)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if delete: bds_path = os.path.join('bds', 'del', scene_name + '.obj')
        else: bds_path = os.path.join('bds', 'add', scene_name + '.obj')
        bd_box = torchMesh(bds_path)

        if sds: out_path = os.path.join('inpaint_sds', 'seq', scene_name)
        else: out_path = os.path.join('inpaint', 'seq', scene_name)
        if not os.path.exists(out_path): os.makedirs(out_path)

        actual_scene = scene_name.split('_')[0]
        cam_param = cam_dict[actual_scene]

        # modes = ['x1', 'x2', 'y1', 'y2', 'xy11', 'xy12', 'xy21', 'xy22']
        modes = ['x1', 'x2']
        for mode in modes:
            render_set(out_path, mode, scene.loaded_iter,
                       scene.getSeqCameras(bd_box, mode=mode, r_scale=cam_param['r_scale'],
                                           k_lift=cam_param['k_lift'], k_bias=cam_param['k_bias'],
                                           view_range=cam_param['view_range']),
                       gaussians, pipeline, background, bd_box)
        if not sds:
            render_set(out_path, "bds_train", scene.loaded_iter,
                       scene.getTrainCameras(),
                       gaussians, pipeline, background, bd_box)
        '''
        if not sds:
            sds_cams = scene.getSeqCameras(bd_box, mode='x1', r_scale=cam_param['r_scale'], k_lift=cam_param['k_lift'],
                                           k_bias=cam_param['k_bias'], view_range=cam_param['view_range'],
                                           change_size=False)
            sds_cams.reverse()
            sds_cams += scene.getSeqCameras(bd_box, mode='x2', r_scale=cam_param['r_scale'], k_lift=cam_param['k_lift'],
                                            k_bias=cam_param['k_bias'], view_range=cam_param['view_range'],
                                            change_size=False)[1:]

            render_set(out_path, 'sds', scene.loaded_iter, sds_cams,
                       gaussians, pipeline, background, bd_box, view_id=False)
        '''

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--sds", action="store_true")
    parser.add_argument("--scene_id", default=None, type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.scene_id)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, args.delete, args.sds, args.scene_id)