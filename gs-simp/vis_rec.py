import cv2
import numpy as np
import os

# scenes = ['bicycle', 'garden', 'kitchen', 'stump']
root = 'vis/vis_video/inpainted'
scenes = None
if scenes is None:
    scenes = os.listdir(root)

img_root = f'{root}/{scenes[0]}/ours_30000/renders'
img0 = cv2.imread(f'{img_root}/00000.png')
h, w = img0.shape[:2]

fps = 20
n_frames = 120
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'vis/rec.avi', fourcc, fps, (w,  h))

for scene in scenes:

    frames = []

    for i in range(n_frames):
        v_id = '%05d' % (i)
        frames.append(cv2.imread(f'{root}/{scene}/ours_30000/renders/{v_id}.png'))

    # frames.reverse()

    for frame in frames:
        out.write(frame)

out.release()
