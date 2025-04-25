from metrics import *
import os
import json
import numpy as np
import sys
from helpers import *

exp = 'ours'

pd_sfxs = {'ours': '/ours_30000/renders',}
gt_sfxs = {'ours': '/ours_30000/renders',}

pd_sfx = pd_sfxs[exp]
gt_sfx = gt_sfxs[exp]
print(pd_sfx, gt_sfx)


n_frame = 10
skips = []
root = f'/home/zhonghongliang/ndpro/metric/vis/cmp/{exp}'
out_path = f'/home/zhonghongliang/ndpro/metric/results/1124_1/cmp_{exp}.json'

musiq = MUSIQ()

scenes = os.listdir(f'{root}/inpainted')
results = {'text': {}, 'directional': {}, 'musiq': {}}

for scene in scenes:

    if scene in skips: continue

    prompt = text_dict[scene]
    origin_prompt = text_origin[scene.split('_')[0]]

    pd_path = f'{root}/inpainted/{scene}{pd_sfx}'
    gt_path = f'{root}/src/{scene}{gt_sfx}'

    scores = {'text': [], 'directional': [], 'musiq': []}

    for idx in range(n_frame):

        v_id = "{0:05d}".format(idx)

        f_pd = f'{pd_path}/{v_id}.png'
        f_gt = f'{gt_path}/{v_id}.png'

        scores['text'].append(text_img_sim(f_pd, prompt))
        scores['directional'].append(directional_sim(f_gt, f_pd, origin_prompt, prompt))
        scores['musiq'].append(musiq(f_pd))

    results['text'][scene] = float(np.mean(scores['text']))
    results['directional'][scene] = float(np.mean(scores['directional']))
    results['musiq'][scene] = float(np.mean(scores['musiq']))

    print(scene, ' ',
          "%.6f" % results['text'][scene],
          "%.6f" % results['directional'][scene],
          "%.6f" % results['musiq'][scene],)

print('# AVG Text: ', np.mean(list(results['text'].values())))
print('# AVG Directional: ', np.mean(list(results['directional'].values())))
print('# AVG MUSIQ: ', np.mean(list(results['musiq'].values())))

with open(out_path, 'w') as f:
    json.dump(results, f)


