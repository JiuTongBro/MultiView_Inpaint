import numpy as np
from plyfile import PlyData, PlyElement
import torch
import os
from utils.bounding import *


# scenes = ['1', '2', '3', '3b', '4', '7', '9', '10', '12', 'book', 'trash']
scenes = ['bicycle', 'bonsai', 'garden', 'kitchen', 'room', 'stump']
# scenes = ['counter',]
max_sh_degree = 0
root_dir = 'output'


def save_ply(comps, path):

    xyz, features_dc, features_rest, opacity, scaling, rotation = comps
    print(xyz.size(), features_dc.size(), features_rest.size(), opacity.size(), scaling.size(), rotation.size())

    # construct_list_of_attributes()
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(features_dc.shape[1] * features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(features_rest.shape[1] * features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))

    xyz = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scale = scaling.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    _xyz = torch.tensor(xyz, dtype=torch.float, device="cuda") # n, 3
    _features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    _features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    _opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
    _scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
    _rotation = torch.tensor(rots, dtype=torch.float, device="cuda")

    print(_xyz.size(), _features_dc.size(), _features_rest.size(), _opacity.size(), _scaling.size(), _rotation.size())

    return _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation

for scene in scenes:

    print(scene)

    bds_path = os.path.join('bds', 'del', scene + '.obj')
    if os.path.exists(bds_path):
        bd_box = torchMesh(bds_path)

        ply_path = os.path.join(root_dir, scene, 'point_cloud', 'iteration_30000', 'point_cloud.ply')
        ply_comps = load_ply(ply_path)
        xyz, features_dc, features_rest, opacity, scaling, rotation = ply_comps

        n = xyz.size()[0]
        pos_d = torch.tensor([[1., 0., 0.]], dtype=torch.float, device="cuda").repeat((n, 1))
        neg_d = torch.tensor([[-1., 0., 0.]], dtype=torch.float, device="cuda").repeat((n, 1))
        _, pos_t, _, _ = bd_box.intersect(xyz, pos_d)  # n, 1
        _, neg_t, _, _ = bd_box.intersect(xyz, neg_d)  # n, 1
        inside = ((pos_t > 0.) & (neg_t > 0.))[..., 0]

        xyz = xyz[~inside]
        features_dc = features_dc[~inside]
        features_rest = features_rest[~inside]
        opacity = opacity[~inside]
        scaling = scaling[~inside]
        rotation = rotation[~inside]
        comps = (xyz, features_dc, features_rest, opacity, scaling, rotation)
    else:
        print('Skip ', scene)
        ply_path = os.path.join(root_dir, scene, 'point_cloud', 'iteration_30000', 'point_cloud.ply')
        ply_comps = load_ply(ply_path)
        xyz, features_dc, features_rest, opacity, scaling, rotation = ply_comps

        comps = (xyz, features_dc, features_rest, opacity, scaling, rotation)

    out_dir = os.path.join(root_dir, scene, 'point_cloud', 'del')
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    out_path = os.path.join(out_dir, 'point_cloud.ply')
    save_ply(comps, out_path)




