import os
import cv2
from SegTracker import SegTracker
from model_args import aot_args, sam_args, segtracker_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
import sys

n_frames = 14
scene_name = sys.argv[1]
obj_name = sys.argv[2]
mode = sys.argv[3] # bds, bds_reverse
ctrl_id = int(sys.argv[4]) # bds, bds_reverse
root = '/home/zhonghongliang/ndpro/gs-simp/inpaint'

if ctrl_id >= 0: ctrl_sfx = f'ctrl_{ctrl_id}/'
else: ctrl_sfx = ''

io_args = {
    'input_dir': f'{root}/inpainted/{scene_name}/{ctrl_sfx}{mode}',
    'output_dir': f'{root}/sam_mask/{scene_name}/{ctrl_sfx}{mode}', # save pred masks
    'output_video': f'{root}/sam_mask/{scene_name}_{ctrl_id}_{mode}.mp4',  # save pred masks
}

print('# OBJ NAME: ', obj_name)


def save_prediction(pred_mask, output_dir, file_name):
    pred_mask = np.where(pred_mask > 0., 255, 0)
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    # save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir, file_name))


def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)


def draw_mask(img, mask, alpha=0.7, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id * 3:id * 3 + 3]
            else:
                color = [0, 0, 0]
            foreground = img * (1 - alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask != 0)
        countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours, :] = 0

    return img_mask.astype(img.dtype)


# choose good parameters in sam_args based on the first frame segmentation result
# other arguments can be modified in model_args.py
# note the object number limit is 255 by default, which requires < 10GB GPU memory with amp
sam_args['generator_args'] = {
        'points_per_side': 30,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    }

grounding_caption = obj_name
box_threshold, text_threshold, box_size_threshold, reset_image = 0.35, 0.5, 0.5, True

frames = []
for i in range(n_frames):
    img = cv2.imread(os.path.join(io_args['input_dir'], '{0:02d}'.format(i) + ".png"))
    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

frame_idx = 0
segtracker = SegTracker(segtracker_args,sam_args,aot_args)
segtracker.restart_tracker()
with torch.cuda.amp.autocast():
    for frame in frames:
        pred_mask, annotated_frame = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold)
        torch.cuda.empty_cache()
        obj_ids = np.unique(pred_mask)
        obj_ids = obj_ids[obj_ids!=0]
        print("processed frame {}, obj_num {}".format(frame_idx,len(obj_ids)),end='\n')
        break
    init_res = draw_mask(annotated_frame, pred_mask, id_countour=False)

    del segtracker
    torch.cuda.empty_cache()
    gc.collect()

# For every sam_gap frames, we use SAM to find new objects and add them for tracking
# larger sam_gap is faster but may not spot new objects in time
segtracker_args = {
    'sam_gap': 49,  # the interval to run sam to segment new objects
    'min_area': 200,  # minimal mask area to add a new mask as a new object
    'max_obj_num': 255,  # maximal object number to track in a video
    'min_new_obj_iou': 0.8,  # the area of a new object in the background should > 80%
}

# output masks
output_dir = io_args['output_dir']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
pred_list = []

torch.cuda.empty_cache()
gc.collect()
sam_gap = segtracker_args['sam_gap']
frame_idx = 0
segtracker = SegTracker(segtracker_args, sam_args, aot_args)
segtracker.restart_tracker()

with torch.cuda.amp.autocast():
    for frame in frames:
        if frame_idx == 0:
            pred_mask, _ = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold,
                                                     box_size_threshold, reset_image)
            torch.cuda.empty_cache()
            gc.collect()
            segtracker.add_reference(frame, pred_mask)
        else:
            pred_mask = segtracker.track(frame, update_memory=True)

        torch.cuda.empty_cache()
        gc.collect()

        save_prediction(pred_mask, output_dir, '{0:02d}'.format(frame_idx) + ".png")
        pred_list.append(pred_mask)

        print("processed frame {}, obj_num {}".format(frame_idx, segtracker.get_obj_num()), end='\r')
        frame_idx += 1
    print('\nfinished')

fps = 2
height, width = frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

frame_idx = 0
out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))
for frame in frames:
    pred_mask = pred_list[frame_idx]
    masked_frame = draw_mask(frame, pred_mask)
    # masked_frame = masked_pred_list[frame_idx]
    masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
    out.write(masked_frame)
    print('frame {} writed'.format(frame_idx),end='\r')
    frame_idx += 1
out.release()
print("\n{} saved".format(io_args['output_video']))
print('\nfinished')

# manually release memory (after cuda out of memory)
del segtracker
torch.cuda.empty_cache()
gc.collect()
