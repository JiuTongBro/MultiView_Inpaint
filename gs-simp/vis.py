import cv2
import numpy as np
import sys

root = '/home/zhonghongliang/ndpro/gs-simp/vis/vis_video/inpainted/bicycle_doll_ctrl_27/ours_30000/renders'
n_frame = 27



frames = []

fps = 4
img0 = cv2.imread(f'{root}/00000.png')
h, w = img0.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'vis/vis.avi', fourcc, fps, (w,  h))

for i in range(n_frame):
    v_id = '%05d' % (i)
    frames.append(cv2.imread(f'{root}/{v_id}.png'))

for frame in frames: out.write(frame)

out.release()
