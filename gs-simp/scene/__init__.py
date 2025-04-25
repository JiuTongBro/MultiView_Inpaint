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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel, InpaintGaussianModel, SDGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.helpers import *
import copy
from utils.bounding import *
from PIL import Image
from utils.camera_utils import *


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], do_delete=False, test=False, n_mode=None, sds=False, ctrl_id='-1'):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.scene_name = os.path.basename(self.model_path)
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args

        if n_mode is None:
            load_path = self.model_path
        else:
            ctrl_id_ = int(ctrl_id)
            if ctrl_id_ >= 0:
                load_path = os.path.join(self.model_path, f'ctrl_{ctrl_id_}')
            else:
                load_path = os.path.join(self.model_path, str(n_mode))

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(load_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        actual_scene = self.scene_name.split('_')[0]
        if actual_scene in ['1', '2', '3', '3b', '4', '7', '9', '10', '12', 'book', 'trash']:
            args.resolution, self.args.resolution = 4, 4
            print(f'### Spin-NeRF Dataset, transfer to {args.resolution}x.')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            if sds:
                ply_path = os.path.join(load_path, "point_cloud", "iteration_" + str(self.loaded_iter),
                                        "point_cloud.ply")
            else:
                ply_path = os.path.join(load_path, "point_cloud", "add", "point_cloud.ply")
                print(ply_path)
                print(test, do_delete, os.path.exists(ply_path))
                if test or do_delete or not os.path.exists(ply_path):
                    ply_path = os.path.join(load_path, "point_cloud", "del", "point_cloud.ply")
                    if test or do_delete or not os.path.exists(ply_path):
                        ply_path = os.path.join(load_path, "point_cloud", "iteration_" + str(self.loaded_iter),
                                                "point_cloud.ply")
            print('# Loaded ply from: ', ply_path)
            self.gaussians.load_ply(ply_path)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, out_root=None):
        if out_root is None: out_root = self.model_path
        point_cloud_path = os.path.join(out_root, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getSeqCameras(self, bd_box, mode='x1', frames=14, view_range=np.pi/3., y_range=np.pi/12.,
                      train_scale=1.0, r_scale=1., k_lift=0., k_bias=0., change_size=True, new_size=[512, 384]):
        train_cams = self.train_cameras[train_scale]
        actual_scene = self.scene_name.split('_')[0]
        for view in train_cams:
            if view.image_name == front_dict[actual_scene]:
                front_view = view
                break

        front_c2w = front_view.camera_to_world[:3]
        makeup = torch.tensor([0., 0., 0., 1.])[None, :].to(front_c2w.device)
        front_pose = front_c2w[:, 3]
        front_y = normalization(front_c2w[:, 1])[None, :]  # 1,3
        box_axes = torch.cat([bd_box.axes, -bd_box.axes], dim=0)
        box_axes = normalization(box_axes, dim=-1)


        cos = torch.sum(front_y * box_axes, dim=-1)
        y_ind = torch.argmax(cos)
        y_axis = box_axes[y_ind]

        box_center = bd_box.center[0]
        front2center = box_center - front_pose
        r = torch.norm(front2center, p=2, dim=-1, keepdim=True)
        scaled_r = r * r_scale

        norm_f2c = normalization(front2center)
        x_axis = normalization(torch.cross(y_axis, norm_f2c))
        z_axis = normalization(torch.cross(x_axis, y_axis))

        views = []
        for v_i in range(frames):

            # 1 to left and up

            if mode in ['x1', 'x2']: # horizon

                angle = view_range * float(v_i) / float(frames)
                if mode == 'x1': angle = - angle
                angle = angle + k_bias
                pose = box_center - z_axis * scaled_r * np.cos(angle) + x_axis * scaled_r * np.sin(angle) \
                       - y_axis * scaled_r * np.sin(k_lift)
                       # np.tan(np.arctan(np.sin(k_lift))), theta=np.arctan(np.sin(k_lift))
                z_vec = normalization(box_center - pose)
                x_vec = normalization(torch.cross(y_axis, z_vec))
                y_vec = normalization(torch.cross(z_vec, x_vec))
                c2w = torch.stack([x_vec, y_vec, z_vec, pose], dim=1)
                c2w = torch.cat([c2w, makeup], dim=0)

            elif mode in ['y1', 'y2']:

                angle = y_range * float(v_i) / float(frames)
                if mode == 'y1': angle = - angle
                pose = box_center - z_axis * scaled_r * np.cos(angle) + y_axis * scaled_r * np.sin(angle) \
                       - y_axis * scaled_r * np.sin(k_lift)
                z_vec = normalization(box_center - pose)
                y_vec = normalization(torch.cross(z_vec, x_axis))
                x_vec = normalization(torch.cross(y_vec, z_vec))

                c2w = torch.stack([x_vec, y_vec, z_vec, pose], dim=1)
                c2w = torch.cat([c2w, makeup], dim=0)

            view = copy.deepcopy(front_view)
            if change_size:
                view.update_attr('{0:02d}'.format(v_i), c2w, change_size=change_size,
                                 image_width=new_size[1], image_height=new_size[0],)
            else: view.update_attr('{0:02d}'.format(v_i), c2w, change_size=change_size)
            views.append(view)

        return views

    def getInpaintCameras(self, n_mode, ctrl_id='-1', frames=14, train_scale=1.0):

        mode_list = ['x2', 'x1', 'y1', 'y2', 'xy21', 'xy22', 'xy11', 'xy12']
        used_modes = mode_list[:n_mode]

        actual_scene = self.scene_name.split('_')[0]
        train_cams = self.train_cameras[train_scale]
        for view in train_cams:
            if view.image_name == front_dict[actual_scene]:
                front_view = view
                break

        def get_seq_data(mode):

            ctrl_id_ = int(ctrl_id)
            if ctrl_id_ >= 0 :
                seq_root = os.path.join('inpaint', 'seq', self.scene_name, mode, 'ours_30000')
                print('\n # Using Sam Mask! #')
                mask_root = os.path.join('inpaint', 'sam_mask', self.scene_name, f'ctrl_{ctrl_id_}', mode)
                # print('\n # Using Box Mask! #')
                # mask_root = os.path.join(seq_root, 'mask')
                inpainted_root = os.path.join('inpaint', 'inpainted', self.scene_name, f'ctrl_{ctrl_id_}', mode)
            else:
                seq_root = os.path.join('inpaint', 'seq', self.scene_name, mode, 'ours_30000')
                print('\n # Using Sam Mask! #')
                mask_root = os.path.join('inpaint', 'sam_mask', self.scene_name, mode)
                # print('\n # Using Box Mask! #')
                # mask_root = os.path.join(seq_root, 'mask')
                inpainted_root = os.path.join('inpaint', 'inpainted', self.scene_name, mode)

            views = []
            poses = np.load(os.path.join(seq_root, 'poses.npy'))
            for index in range(frames):
                v_id = '%02d' % (index)
                v_pose = torch.tensor(poses[index], dtype=torch.float).to(front_view.data_device)
                mask = Image.open(os.path.join(mask_root, f'{v_id}.png')).convert("L")
                mask = PILtoTorch(mask)
                raw_img = Image.open(os.path.join(seq_root, 'renders', f'{v_id}.png')).convert("RGB")
                if os.path.exists(inpainted_root):
                    image = Image.open(os.path.join(inpainted_root, f'{v_id}.png')).convert("RGB")
                    image, raw_img = PILtoTorch(image), PILtoTorch(raw_img)
                    new_img = image * mask + raw_img * (1. - mask)
                    # new_img = image
                else: new_img = PILtoTorch(raw_img)

                view = copy.deepcopy(front_view)
                view.update_attr(v_id, v_pose, original_image=new_img, mask=mask)
                views.append(view)

            return views

        view_list = get_seq_data(used_modes[0])
        for used_mode in used_modes[1:]:
            view_list += get_seq_data(used_mode)[1:]

        return view_list


    def getSDSCameras(self, bd_box, view_range=np.pi/3., shuffle=True):
        cos_thres = np.cos(view_range)
        box_center = bd_box.center[0]
        train_mask_path = os.path.join('inpaint', 'seq', self.scene_name, 'bds_train', 'ours_30000')
        pose_path = os.path.join('inpaint', 'seq', self.scene_name, 'x1', 'ours_30000', 'poses.npy')
        key_pose = torch.from_numpy(np.load(pose_path)[0]).to(box_center.device)

        cam2center = box_center - key_pose[:3, 3]
        r = torch.norm(cam2center, p=2, dim=-1)
        front2center = cam2center / r
        train_cams = self.getTrainCameras()

        new_train_cams = []
        for train_cam in train_cams:

            cam2center = box_center - train_cam.camera_center
            r = torch.norm(cam2center, p=2, dim=-1)
            cam2center = cam2center / r
            cos = torch.sum(cam2center * front2center)

            if cos > cos_thres:
                image = Image.open(os.path.join(train_mask_path, 'renders', f'{train_cam.image_name}.png')).convert("RGB")
                mask = Image.open(os.path.join(train_mask_path, 'mask', f'{train_cam.image_name}.png')).convert("L")
                mask, image = PILtoTorch(mask), PILtoTorch(image)

                if torch.max(mask) > 0.:
                    view = copy.deepcopy(train_cam)
                    view.update_img_mask(image, mask)
                    new_train_cams.append(view)

        random.shuffle(new_train_cams)
        print('# Total Inpaint Training Cams: ', len(new_train_cams))
        return new_train_cams


    def VisInpaintCameras(self, bd_box, frames=10, view_range=np.pi/3., train_scale=1.0,
                          r_scale=1., k_lift=0., k_bias=0, change_size=True):
        train_cams = self.train_cameras[train_scale]
        actual_scene = self.scene_name.split('_')[0]
        for view in train_cams:
            if view.image_name == front_dict[actual_scene]:
                front_view = view
                break

        front_c2w = front_view.camera_to_world[:3]
        makeup = torch.tensor([0., 0., 0., 1.])[None, :].to(front_c2w.device)
        front_pose = front_c2w[:, 3]
        front_y = normalization(front_c2w[:, 1])[None, :]  # 1,3
        box_axes = torch.cat([bd_box.axes, -bd_box.axes], dim=0)
        box_axes = normalization(box_axes, dim=-1)

        cos = torch.sum(front_y * box_axes, dim=-1)
        y_ind = torch.argmax(cos)
        y_axis = box_axes[y_ind]

        box_center = bd_box.center[0]
        front2center = box_center - front_pose
        r = torch.norm(front2center, p=2, dim=-1, keepdim=True)
        scaled_r = r * r_scale

        norm_f2c = normalization(front2center)
        x_axis = normalization(torch.cross(y_axis, norm_f2c))
        z_axis = normalization(torch.cross(x_axis, y_axis))

        views = []
        for v_i in range(frames):

            angle = view_range * ((float(v_i) / float(frames)) * 2. - 1.)
            angle = angle + k_bias

            pose = box_center - z_axis * scaled_r * np.cos(angle) + x_axis * scaled_r * np.sin(angle) \
                   - y_axis * scaled_r * np.sin(k_lift)
            z_vec = normalization(box_center - pose)
            x_vec = normalization(torch.cross(y_axis, z_vec))
            y_vec = normalization(torch.cross(z_vec, x_vec))
            c2w = torch.stack([x_vec, y_vec, z_vec, pose], dim=1)
            c2w = torch.cat([c2w, makeup], dim=0)

            view = copy.deepcopy(front_view)
            view.update_attr('{0:02d}'.format(v_i), c2w, change_size=change_size)
            views.append(view)

        return views

    def VisTrainCameras(self):
        scene_info = sceneLoadTypeCallbacks["Colmap"](self.args.source_path, self.args.images, self.args.eval)
        return cameraList_from_camInfos(scene_info.train_cameras, 1., self.args)


class InpaintScene(Scene):

    gaussians : InpaintGaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):

        self.model_path = args.model_path
        scene_id = os.path.basename(self.model_path)
        original_scene = scene_id.split('_')[0]
        loaded_path = os.path.join(os.path.dirname(self.model_path), original_scene)

        if 'output_sds' in loaded_path:
            self.loaded_path = loaded_path.replace('output_sds', 'output')
        elif 'output_rec' in loaded_path:
            self.loaded_path = loaded_path.replace('output_rec', 'output')
        elif 'outdemo_sds' in loaded_path:
            self.loaded_path = loaded_path.replace('outdemo_sds', 'output')
        self.scene_name = os.path.basename(self.model_path)
        self.gaussians = gaussians
        self.args = args

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),                                                       'wb') as dest_file:
            dest_file.write(src_file.read())
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        actual_scene = self.scene_name.split('_')[0]
        if actual_scene in ['1', '2', '3', '3b', '4', '7', '9', '10', '12', 'book', 'trash']:
            args.resolution, self.args.resolution = 4, 4
            print(f'### Spin-NeRF Dataset, transfer to {args.resolution}x.')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        bds_path = os.path.join('bds', 'add', self.scene_name + '.obj')
        bd_box = torchMesh(bds_path)
        ply_path = os.path.join(self.loaded_path, "point_cloud", "del", "point_cloud.ply")
        self.gaussians.load_sd_ply(ply_path, bd_box)

    def getInpaintTrainCameras(self, n_mode=2, ctrl_id='-1', frames=14, train_scale=1.0):

        train_mask_path = os.path.join('inpaint', 'seq', self.scene_name, 'bds_train', 'ours_30000')
        seq_cams = self.getInpaintCameras(n_mode, ctrl_id, frames, train_scale)
        train_cams = self.getTrainCameras()
        new_train_cams = []
        for train_cam in train_cams:
            image = Image.open(os.path.join(train_mask_path, 'renders', f'{train_cam.image_name}.png')).convert("RGB")
            mask = Image.open(os.path.join(train_mask_path, 'mask', f'{train_cam.image_name}.png')).convert("L")
            mask, image = PILtoTorch(mask), PILtoTorch(image)
            view = copy.deepcopy(train_cam)
            view.update_img_mask(image, mask)
            new_train_cams.append(view)

        n_train_cam = len(new_train_cams)
        n_seq_cam = n_mode * frames

        if n_seq_cam >= n_train_cam * 2:
            k_repeat = n_seq_cam // n_train_cam
            new_trains = []
            for k in range(k_repeat):
                for train_cam in new_train_cams:
                    view = copy.deepcopy(train_cam)
                    new_trains.append(view)
            new_cams = seq_cams + new_trains
        elif n_train_cam >= n_seq_cam * 2:
            k_repeat = n_train_cam // n_seq_cam
            new_seq_cams = []
            for k in range(k_repeat):
                for seq_cam in seq_cams:
                    view = copy.deepcopy(seq_cam)
                    new_seq_cams.append(view)
            new_cams = new_seq_cams + new_train_cams
        else:
            new_cams = seq_cams + new_train_cams

        random.shuffle(new_cams)
        print('# Total Inpaint Training Cams: ', len(new_cams))
        return new_cams













