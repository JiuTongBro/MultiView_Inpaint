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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, bd_box, inpainted):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gt_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gts")

    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    if not inpainted:
        mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask")
        makedirs(mask_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        outputs = render(view, gaussians, pipeline, background)
        rendering, depth = outputs["render"], outputs["depth"]
        gt = view.original_image[0:3, :, :]

        if not inpainted:
            rays_o, rays_d = get_rays(view)
            _, inter_t, _, _ = bd_box.intersect(rays_o, rays_d)  # n, 1
            inter_t = inter_t.view(int(view.image_height), int(view.image_width))[None, ...]  # 1, h, w
            inter_mask = (inter_t > 0.) & ((inter_t < depth) | (depth == 15.))
            inter_mask = inter_mask.float()

            torchvision.utils.save_image(inter_mask, os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams,
                skip_train : bool, skip_test : bool, n_mode, inpainted, scene_id, ctrl_id):

    vis_inpaint_view = False

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        
        if inpainted: n_mode=n_mode
        else: n_mode=None

        # n_mode = None

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, n_mode=n_mode, ctrl_id=ctrl_id)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        scene_name = os.path.basename(dataset.model_path)
        if inpainted:
            bds_path = os.path.join('bds', 'add', scene_name + '.obj')
        else: bds_path = os.path.join('bds', 'add', scene_id + '.obj')
        bd_box = torchMesh(bds_path)

        actual_scene = scene_name.split('_')[0]
        cam_param = vis_dict[actual_scene]
        # cam_param = cam_dict[actual_scene]

        if vis_inpaint_view: prefix = '_inpaint'
        else: prefix = ''

        if inpainted: out_path = 'vis/vis_video/inpainted' + prefix
        else: out_path = 'vis/vis_video/src' + prefix
        if not os.path.exists(out_path): os.makedirs(out_path)

        '''
        render_set(dataset.model_path, "inpaint", scene.loaded_iter,
                   scene.getTrainCameras(), gaussians, pipeline, background)
        '''

        if vis_inpaint_view:

            n_frame = 14
            vis_cameras = scene.getInpaintCameras(n_mode, ctrl_id)
            vis_cameras1 = vis_cameras[:n_frame]
            vis_cameras1.reverse()
            vis_cameras2 = vis_cameras[n_frame:]
            vis_cameras = vis_cameras1 + vis_cameras2

        else:
        
            vis_cameras = scene.VisInpaintCameras(bd_box, r_scale=cam_param['r_scale'],
                                               k_lift=cam_param['k_lift'],
                                               view_range=cam_param['view_range'],
                                               change_size=True)

            # vis video:
            # garden_gnome, np.pi * 2. / 9., b = np.pi / 36.
            # bicycle_doll, np.pi * 19 / 72., b = - np.pi * 3 / 72.
                                           



        if inpainted:
            ctrl_id_ = int(ctrl_id)
            if ctrl_id_ >= 0:
                out_name = f'{scene_name}_ctrl_{ctrl_id_}'
            else: out_name = scene_name
        else: out_name = scene_id
        render_set(out_path, out_name, scene.loaded_iter, vis_cameras,
                   gaussians, pipeline, background, bd_box, inpainted)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--n_mode', type=int, default=2)
    parser.add_argument('--inpainted', action="store_true")
    parser.add_argument("--scene_id", default='', type=str)
    parser.add_argument("--ctrl_id", default='-1', type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train,
                args.skip_test, args.n_mode, args.inpainted, args.scene_id, args.ctrl_id)