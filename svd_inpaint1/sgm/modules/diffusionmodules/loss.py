from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser

from einops import rearrange, repeat


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")


class InpaintDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
        additional_cond_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)
        self.additional_cond_keys = set(additional_cond_keys)

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor,
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:

        cond = conditioner(batch)

        if 'num_video_frames' in batch.keys():
            for k in ["crossattn", "concat"]:
                if k in cond.keys():
                    cond[k] = repeat(cond[k], "b ... -> (b t) ...", t=batch['num_video_frames'])

        cond_keys = self.additional_cond_keys.intersection(batch)
        for k in cond_keys:
            if k in ['crossattn_scale', 'concat_scale', 'prev_frame']:
                cond[k] = repeat(batch[k], "b ... -> (b t) ...", t=batch['num_video_frames'])
            else:
                cond[k] = batch[k]

        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:

        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        if 'num_video_frames' in batch.keys():
            b = input.shape[0] // batch['num_video_frames']
            sigmas = self.sigma_sampler(b).to(input)
            sigmas = repeat(sigmas, "b ... -> b t ...", t=batch['num_video_frames']).contiguous()
            sigmas = rearrange(sigmas, "b t ... -> (b t) ...", t=batch['num_video_frames'])
        else:
            sigmas = self.sigma_sampler(input.shape[0]).to(input)

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)

        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        # model_output = model_output * mask + input * (1. - mask)

        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)


    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")


class InpaintDiffusionLoss2(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
        additional_cond_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)
        self.additional_cond_keys = set(additional_cond_keys)

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor,
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:

        cond = conditioner(batch)

        if 'num_video_frames' in batch.keys():
            for k in ["crossattn", "concat"]:
                if k in cond.keys():
                    cond[k] = repeat(cond[k], "b ... -> (b t) ...", t=batch['num_video_frames'])

        cond_keys = self.additional_cond_keys.intersection(batch)
        for k in cond_keys:
            if k in ['crossattn_scale', 'concat_scale', 'prev_frame']:
                cond[k] = repeat(batch[k], "b ... -> (b t) ...", t=batch['num_video_frames'])
            else:
                cond[k] = batch[k]

        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:

        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        if 'num_video_frames' in batch.keys():
            b = input.shape[0] // batch['num_video_frames']
            sigmas = self.sigma_sampler(b).to(input)
            sigmas = repeat(sigmas, "b ... -> b t ...", t=batch['num_video_frames']).contiguous()
            sigmas = rearrange(sigmas, "b t ... -> (b t) ...", t=batch['num_video_frames'])
        else:
            sigmas = self.sigma_sampler(input.shape[0]).to(input)

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)

        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        # model_output = model_output * mask + input * (1. - mask)

        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        return self.get_loss(model_output, input, w, batch)


    def get_loss(self, model_output, target, w_, batch, mean_warp=False):
        if self.loss_type == "l2":
            loss = torch.mean(
                (w_ * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            loss = torch.mean(
                (w_ * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

        num_frames, num_channels, h, w = target.shape[0], target.shape[1], target.shape[2], target.shape[3],
        hit_map, frames_ind, = batch['hit_map'], batch['uv_ind']
        prev_frames = model_output[:num_frames - 1]
        flat_frames = prev_frames.view(num_frames - 1, num_channels, -1)
        flat_ind = frames_ind.view(num_frames - 1, num_channels, -1)
        flat_projected = torch.gather(flat_frames, dim=-1, index=flat_ind)
        projected = flat_projected.view(num_frames - 1, num_channels, h, w)

        warp_error = (projected - model_output[1:]) * hit_map[:, None, :, :]
        if mean_warp:
            if self.loss_type == "l2":
                warp_loss = w_ * torch.mean(warp_error ** 2, dim=0, keepdim=True)
            elif self.loss_type == "l1":
                warp_loss = w_ * torch.mean(warp_error.abs(), dim=0, keepdim=True)
            loss += torch.mean(warp_loss.reshape(num_frames, -1), dim=1)
        else:
            if self.loss_type == "l2":
                warp_loss = w_[1:] * (warp_error ** 2)
            elif self.loss_type == "l1":
                warp_loss = w_[1:] * (warp_error.abs())
            loss[1:] += torch.mean(warp_loss.reshape(num_frames - 1, -1), dim=1)

        return loss