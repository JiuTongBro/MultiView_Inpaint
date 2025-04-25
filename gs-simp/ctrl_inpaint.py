from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import os
from scene.helpers import *
from PIL import Image

delete = False

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-depth")
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

n_samples = 200
root = 'inpaint'

if delete:
    scenes = ['garden',]
    prompts = {'garden': 'an empty table',}
else:
    scenes = ['bicycle_doll',]
    if scenes is None: scenes = os.listdir('inpaint/seq')
    prompts = text_dict

for scene in scenes:

    print(scene)

    scene_dir = os.path.join(root, 'seq', scene, 'x1', 'ours_30000')
    depth_path = os.path.join(root, 'depth', scene, 'x1', '00.png')
    img_path = os.path.join(scene_dir, 'renders', '00.png')
    mask_path = os.path.join(scene_dir, 'mask', '00.png')

    out_path = os.path.join(root, 'ctrl', scene,)
    if not os.path.exists(out_path): os.makedirs(out_path)

    image = Image.open(img_path).convert('RGB')
    w, h = image.size
    image = image.resize((512, 512))
    mask_image = Image.open(mask_path).convert('L').resize((512, 512))
    guide_image = Image.open(depth_path).convert('RGB').resize((512, 512))

    text_prompt = prompts[scene]

    for i in range(n_samples):

        out_image = pipe(
            text_prompt,
            image=image,
            control_image=guide_image,
            mask_image=mask_image
        ).images[0]

        out_image = out_image.resize((w, h))
        out_image.save(os.path.join(out_path, f'ctrl_{i}.png'))

