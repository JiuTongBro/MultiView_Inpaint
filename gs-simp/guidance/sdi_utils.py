from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import cv2

from torch.cuda.amp import custom_bwd, custom_fwd


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def prepare_mask_and_masked_image(image, mask):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """

    # Binarize mask
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    masked_image = image * (mask < 0.5)

    return mask, masked_image


class StableDiffusionInpaint(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-inpainting"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-inpainting"
        '''
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')
        '''
        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)
        masked_image_latents = self._encode_vae_image(masked_image, generator=generator)


        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    def train_step(self, text_embeddings, pred_rgb, pred_mask, guidance_scale=100,
                   as_latent=False, grad_scale=1, norm=True,
                   save_guidance_path: Path = None):

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False)
            if norm: latents = latents * 2. - 1.
            pred_mask_64 = F.interpolate(pred_mask, (64, 64), mode='bilinear', align_corners=False)
            pred_mask_64, masked_latents  = prepare_mask_and_masked_image(latents, pred_mask_64)
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            if norm: pred_rgb_512 = pred_rgb_512 * 2. - 1.
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

            pred_mask_512 = F.interpolate(pred_mask, (512, 512), mode='bilinear', align_corners=False)
            pred_mask_512, masked_image = prepare_mask_and_masked_image(pred_rgb_512, pred_mask_512)

            pred_mask_64 = F.interpolate(pred_mask_512, (64, 64), mode='bilinear', align_corners=False)
            masked_latents = self.encode_imgs(masked_image)

        # print('Embedding: ', text_embeddings[:, :2, :2])

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)
        chose_t = t.item()

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            mask_input = torch.cat([pred_mask_64] * 2)
            masked_image_latents = torch.cat([masked_latents] * 2)
            latent_model_input = torch.cat([latent_model_input, mask_input, masked_image_latents], dim=1)

            tt = torch.cat([t] * 2)
            noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        if save_guidance_path:

            with torch.no_grad():

                self.scheduler.set_timesteps(1000)

                if chose_t > 0:

                    steplist = self.scheduler.timesteps[1000 - chose_t:]

                    print('# Denoise Test Image: ', chose_t, steplist[:2], steplist[-2:])

                    for i, t in enumerate(steplist):

                        t = t.to(self.device)

                        # pred noise
                        latent_model_input = torch.cat([latents_noisy] * 2)
                        mask_input = torch.cat([pred_mask_64] * 2)
                        masked_image_latents = torch.cat([masked_latents] * 2)
                        latent_model_input = torch.cat([latent_model_input, mask_input, masked_image_latents], dim=1)

                        tt = torch.cat([t[None, ...]] * 2)
                        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

                        # perform guidance
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy)['prev_sample']

                    print('# Done Denoise.')

                # Img latents -> imgs
                imgs = self.decode_latents(latents_noisy)  # [1, 3, 512, 512]
                # Img to Numpy
                imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
                imgs = (np.clip(imgs[0], 0., 1.) * 255).astype('uint8')

                cv2.imwrite(f"{save_guidance_path}_{str(chose_t)}.png", imgs[..., [2, 1, 0]])

        return loss

    @torch.no_grad()
    def test_step(self, text_embeddings, pred_rgb, pred_mask, chose_t=1000,
                   guidance_scale=100, as_latent=False, grad_scale=1, norm=True,
                   save_guidance_path: Path = None, pure_noise=False):

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False)
            if norm: latents = latents * 2. - 1.
            pred_mask_64 = F.interpolate(pred_mask, (64, 64), mode='bilinear', align_corners=False)
            pred_mask_64, masked_latents  = prepare_mask_and_masked_image(latents, pred_mask_64)
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            if norm: pred_rgb_512 = pred_rgb_512 * 2. - 1.
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

            pred_mask_512 = F.interpolate(pred_mask, (512, 512), mode='bilinear', align_corners=False)
            pred_mask_512, masked_image = prepare_mask_and_masked_image(pred_rgb_512, pred_mask_512)

            pred_mask_64 = F.interpolate(pred_mask_512, (64, 64), mode='bilinear', align_corners=False)
            masked_latents = self.encode_imgs(masked_image)

        # print('Embedding: ', text_embeddings[:, :2, :2])

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.tensor([chose_t,], dtype=torch.long).to(self.device)

        # predict the noise residual with unet, NO grad!
        # add noise
        noise = torch.randn_like(latents)
        if pure_noise: latents_noisy = noise
        else: latents_noisy = self.scheduler.add_noise(latents, noise, t)

        self.scheduler.set_timesteps(1000)

        if chose_t > 0:

            steplist = self.scheduler.timesteps[1000 - chose_t:]

            print('# Denoise Test Image: ', chose_t, steplist[:2], steplist[-2:])

            for i, t in enumerate(steplist):

                t = t.to(self.device)

                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2)
                mask_input = torch.cat([pred_mask_64] * 2)
                masked_image_latents = torch.cat([masked_latents] * 2)
                latent_model_input = torch.cat([latent_model_input, mask_input, masked_image_latents], dim=1)

                tt = torch.cat([t[None, ...]] * 2)
                noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy)['prev_sample']

            print('# Done Denoise.')

        # Img latents -> imgs
        imgs = self.decode_latents(latents_noisy)  # [1, 3, 512, 512]
        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (np.clip(imgs[0], 0., 1.) * 255).astype('uint8')

        return imgs

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                        latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                                  device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2. + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        # Have done before
        # imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts)  # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='1.5', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusionInpaint(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()





