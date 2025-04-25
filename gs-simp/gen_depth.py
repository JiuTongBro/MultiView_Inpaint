import os
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np

# scenes = ['garden', 'bicycle', 'kitchen', 'stump']
scenes = None
if scenes is None:
    scenes = os.listdir('inpaint_sds/seq')
modes = ['x1', 'x2']

base_dir = 'inpaint_sds/seq'
out_dir = 'inpaint/depth'

depth_estimator = pipeline('depth-estimation')

for scene in scenes:

    print(scene)

    for mode in modes:

        scene_dir = os.path.join(base_dir, scene, mode, 'ours_5000', 'renders')

        images = os.listdir(scene_dir)

        out_path = os.path.join(out_dir, scene, mode)
        if not os.path.exists(out_path): os.makedirs(out_path)

        for f_image in images:
            f_img = os.path.join(scene_dir, f_image)
            input_pil = Image.open(f_img).convert('RGB')

            depth_image = depth_estimator(input_pil)['depth']
            depth_image = np.array(depth_image)
            depth_image = depth_image[:, :, None]
            detected_image = np.concatenate(3 * [depth_image], axis=2)

            cv2.imwrite(os.path.join(out_path, f_image), detected_image[..., [2, 1, 0]])


