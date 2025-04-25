import cv2
import os


data_root = '/home/zhonghongliang/ndpro/svd_inpaint1/gs'

# mode_list = ['x1', 'x2', 'y1', 'y2', 'xy11', 'xy12', 'xy21', 'xy22']
mode_list = ['x1', 'x2',]

scenes = os.listdir(os.path.join(data_root, 'ctrl1'))
scenes.sort()

scene_ids = []
for scene in scenes:
    ctrls = os.listdir(os.path.join(data_root, 'ctrl1', scene))
    ctrls.sort()
    for ctrl in ctrls:
        scene_ids.append([scene, ctrl])

n_samples = len(scene_ids) * len(mode_list)
padding = 2

fps = 3
n_h, n_w = 4, 4
n_frame = 14

svd_out_root = 'logs/shiny3'
img_root = f'{svd_out_root}/log_img/test'
out_root = f'{svd_out_root}/out'
if not os.path.exists(out_root):
    os.makedirs(out_root)

vis_dir = f'{svd_out_root}/vis'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

for i in range(n_samples):

    scene, f_ctrl = scene_ids[int(i // len(mode_list))]
    print(scene, f_ctrl)
    mode_id = i % len(mode_list)

    out_dir = os.path.join(out_root, scene, f_ctrl[:-4], mode_list[mode_id])

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_b_id = '%06d' % (i)
    f_img = f'samples_gs-000000_e-000000_b-{img_b_id}.png'

    img_path = os.path.join(img_root, f_img)

    img = cv2.imread(img_path)

    h, w = img.shape[:2]
    h, w = (h - (n_h + 1) * padding) // n_h, (w - (n_w + 1) * padding) // n_w
    print(h, w)

    if mode_id == 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = f'{vis_dir}/{scene}'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        out = cv2.VideoWriter(os.path.join(video_dir, f'{f_ctrl[:-4]}.mp4'), fourcc, fps, (w, h))
        video_frames = []
    else:
        video_frames = video_frames[1:]
        video_frames.reverse()

    for i in range(n_h):
        for j in range(n_w):
            v_i = i * n_w + j
            if v_i >= n_frame: break
            v_id = '%02d' % (v_i)

            h_pad, w_pad = (i + 1) * padding, (j + 1) * padding
            frame = img[i*h+h_pad:(i+1)*h+h_pad, j*w+w_pad:(j+1)*w+w_pad]

            video_frames.append(frame)
            out_f = os.path.join(out_dir, f'{v_id}.png')
            cv2.imwrite(out_f, frame)

    if mode_id == 1:
        for frame in video_frames:
            out.write(frame)
        out.release()

