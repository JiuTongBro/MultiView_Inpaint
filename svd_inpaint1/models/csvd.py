from typing import Any, List, Optional, Union, Dict
import torch
from torch.optim.lr_scheduler import LambdaLR
import os
from matplotlib import pyplot as plt
from PIL import Image
import torchvision
import numpy as np
import itertools

from safetensors.torch import load_file as load_safetensors
from sgm.modules.diffusionmodules.openaimodel import *
from sgm.modules.video_attention import SpatialVideoTransformer
from sgm.util import default, disabled_train, instantiate_from_config, get_obj_from_str, log_txt_as_img
from sgm.modules.diffusionmodules.util import AlphaBlender, zero_module
from sgm.modules.diffusionmodules.video_model import VideoResBlock, VideoUNet
from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.models.diffusion import DiffusionEngine
from sgm.util import count_params
from pytorch_lightning.utilities import rank_zero_only
from sgm.util import isheatmap

import numpy as np


gpu_autocast_kwargs = {
    "enabled": True,  # torch.is_autocast_enabled(),
    "dtype": torch.get_autocast_gpu_dtype(),
    "cache_enabled": torch.is_autocast_cache_enabled(),
}

class ControlledVideoUNet(VideoUNet):
    def forward(
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        context: Optional[th.Tensor] = None,
        y: Optional[th.Tensor] = None,
        time_context: Optional[th.Tensor] = None,
        control: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,
    ):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"

        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x

        for module in self.input_blocks:
            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
            hs.append(h)
    
        h = self.middle_block(
            h,
            emb,
            context=context,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames,
        )

        if control is not None:
            h += control.pop()

        for module in self.output_blocks:
            if control is None:

                h = th.cat([h, hs.pop()], dim=1)

            else:
                # h_control = control.pop()
                # last_h, h_s = h, hs.pop()
                # h = th.cat([h, h_s + h_control], dim=1)
                h = th.cat([h, hs.pop() + control.pop()], dim=1)

            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
        '''
        n_files = len(os.listdir('vis'))
        n_t = n_files // 3

        vis_h = last_h.clone().detach().cpu().numpy()
        vis_hs = h_s.clone().detach().cpu().numpy()
        vis_control = h_control.clone().detach().cpu().numpy()

        np.save(f'vis/h_{n_t}.npy', vis_h)
        np.save(f'vis/hs_{n_t}.npy', vis_hs)
        np.save(f'vis/control_{n_t}.npy', vis_control)
        '''
        h = h.type(x.dtype)

        return self.out(h)



class ControlNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        hint_channels: int,
        num_res_blocks: int,
        attention_resolutions: int,
        dropout: float = 0.0,
        channel_mult: List[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth: Union[List[int], int] = 1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        spatial_transformer_attn_type: str = "softmax",
        video_kernel_size: Union[int, List[int]] = 3,
        use_linear_in_transformer: bool = False,
        adm_in_channels: Optional[int] = None,
        disable_temporal_crossattention: bool = False,
        max_ddpm_temb_period: int = 10000,
    ):
        super().__init__()
        assert context_dim is not None

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.hint_channels = hint_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.context_dim = context_dim
        self.adm_in_channels = adm_in_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )

            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disabled_sa=False,
        ):
            return SpatialVideoTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                time_context_dim=time_context_dim,
                dropout=dropout,
                ff_in=extra_ff_mix_layer,
                use_spatial_context=use_spatial_context,
                merge_strategy=merge_strategy,
                merge_factor=merge_factor,
                checkpoint=use_checkpoint,
                use_linear=use_linear_in_transformer,
                attn_mode=spatial_transformer_attn_type,
                disable_self_attn=disabled_sa,
                disable_temporal_crossattention=disable_temporal_crossattention,
                max_time_embed_period=max_ddpm_temb_period,
            )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_ch,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
        ):
            return VideoResBlock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
            )

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_down=time_downup,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))

                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
            ),
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(
        self,
        x: th.Tensor,
        hint: th.Tensor,
        timesteps: th.Tensor,
        context: Optional[th.Tensor] = None,
        y: Optional[th.Tensor] = None,
        time_context: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,
    ):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        guided_hint = self.input_hint_block(hint, emb, context)
        # np.save('vis/hint.npy', guided_hint.detach().cpu().numpy())

        outs = []

        h = x
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
            if guided_hint is not None:
                h += guided_hint
                guided_hint = None
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(
            h,
            emb,
            context=context,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames,
        )
        outs.append(self.middle_block_out(h, emb, context))

        ########
        '''g1 = self.input_palette_block[0].weight.grad
        g2 = self.out_head.grad
        g3 = self.emb_out[0].weight.grad
        g4 = self.context_out[0].weight.grad
        if g1 is not None:
            g1 = g1.abs().sum()
            g2 = g2.abs().sum()
            g3 = g3.abs().sum()
            g4 = g4.abs().sum()'''
        ########

        return outs
    
    def init_from_ckpt(
        self,
        path: str,
    ) -> None:

        print('# Control Net Resume... #')
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def init_ctrl_from_test(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        prefix = 'control_model.'
        len_prefix = len(prefix)
        new_ctrl = dict()
        for k, v in sd.items():
            if k.startswith(prefix):
                new_k = k[len_prefix:]
                new_ctrl[new_k] = v
        print('\n Load SD_SVD keys!')
        missing, unexpected = self.load_state_dict(new_ctrl, strict=False)

        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )

        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def set_parameters_requires_grad(self):
        self.requires_grad_(True)

    def get_trainable_parameters(self):
        params = []
        for p in self.parameters():
            if p.requires_grad:
                params += [p]

        return params

    def get_blacklist(self):
        return []



class VideoDiffusionEngine(DiffusionEngine):
    def __init__(
            self,
            controlnet_config,
            control_model_path=None,
            init_from_unet=False,
            global_average_pooling=False,
            sd_locked=True,
            drop_first_stage_model=False,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sd_locked = sd_locked
        self.control_model = instantiate_from_config(controlnet_config)
        self.control_scales = [1.0] * 13
        self.global_average_pooling = global_average_pooling

        self.conditioner.embedders[0].eval()
        self.conditioner.embedders[0].requires_grad_(False)
        self.conditioner.embedders[1].eval()
        self.conditioner.embedders[1].requires_grad_(False)
        self.conditioner.embedders[2].eval()
        self.conditioner.embedders[2].requires_grad_(False)
        self.conditioner.embedders[3].eval()
        self.conditioner.embedders[3].requires_grad_(False)
        self.conditioner.embedders[4].eval()
        self.conditioner.embedders[4].requires_grad_(False)

        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)

        if self.sd_locked:
            self.model.diffusion_model.eval()
            self.model.diffusion_model.requires_grad_(False)

        # label_emb shall be true
        self.model.diffusion_model.label_emb.train()
        self.model.diffusion_model.label_emb.requires_grad_(True)

        self.control_model.train()
        self.control_model.set_parameters_requires_grad()

        if self.test_ckpt is not None:
            print('## Testing, reload controlnet from test_ckpt!')
            self.control_model.init_ctrl_from_test(self.test_ckpt)
        else:
            if control_model_path is not None:
                self.control_model.init_from_ckpt(control_model_path)
            if init_from_unet:
                missing, unexpected = self.control_model.load_state_dict(self.model.diffusion_model.state_dict(), strict=False )
                print(f"Restored from UNet {len(missing)} missing and {len(unexpected)} unexpected keys")

        if drop_first_stage_model:
            del self.first_stage_model

    def forward(self, x, batch):
        loss = self.loss_fn(self.apply_model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)

        if x.shape[1] == 3:
            x = self.encode_first_stage(x)

        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def apply_model(
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        cond: Dict,
        time_context: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,

    ):

        B, _, _, _ = x.shape

        cond_concat = cond.get("concat", torch.Tensor([]).type_as(x))
        if 'concat_scale' in cond.keys():
            cond_concat = cond_concat * cond['concat_scale']
        # input_x = torch.cat([x, mask, masked_z, cond_concat], dim=1)
        # input_x_control = torch.cat([x, cond_concat], dim=1)
        input_x = torch.cat([x, cond_concat.type_as(x)], dim=1)
        input_x_control = torch.cat([x, cond_concat.type_as(x)], dim=1)

        context = cond.get('crossattn', None)
        if 'crossattn_scale' in cond.keys():
            context = context * cond['crossattn_scale']

        y = cond.get('vector', None)

        control_hint = cond.get('control_hint', None)
        if 'palette' in cond.keys():
            control_hint = [control_hint, cond['palette']]

        if control_hint is not None:
            controls = self.control_model(
                x=input_x_control,
                hint=control_hint,
                timesteps=timesteps,
                context=context,
                y=y,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            controls = [c * scale for c, scale in zip(controls, self.control_scales)]
            if self.global_average_pooling:
                controls = [torch.mean(c, dim=(2, 3), keepdim=True) for c in controls]
        else:
            controls = None

        out = self.model.diffusion_model(
            x=input_x,
            timesteps=timesteps,
            context=context,
            y=y,
            time_context=time_context,
            control=controls,
            num_video_frames=num_video_frames,
            image_only_indicator=image_only_indicator,
        )

        return out

    '''
    def apply_model_double(
        self,
        cnet_x: th.Tensor,
        unet_x: th.Tensor,
        timesteps: th.Tensor,
        cond: Dict,
        time_context: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,
        
    ):
        B, _, _, _ = unet_x.shape

        cond_concat = cond.get("concat", torch.Tensor([]).type_as(unet_x))
        if 'concat_scale' in cond.keys():
            cond_concat = cond_concat * cond['concat_scale']
        unet_x = torch.cat([unet_x, cond_concat.type_as(unet_x)], dim=1)
        cnet_x = torch.cat([cnet_x, cond_concat.type_as(cnet_x)], dim=1)

        context = cond.get('crossattn', None)
        if 'crossattn_scale' in cond.keys():
            context = context * cond['crossattn_scale']

        y = cond.get('vector', None)

        control_hint = cond.get('control_hint', None)
        if 'palette' in cond.keys():
            control_hint = [control_hint, cond['palette']]

        if control_hint is not None:
            controls = self.control_model(
                x=cnet_x,
                hint=control_hint,
                timesteps=timesteps,
                context=context,
                y=y,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            controls = [c * scale for c, scale in zip(controls, self.control_scales)]
            if self.global_average_pooling:
                controls = [torch.mean(c, dim=(2, 3), keepdim=True) for c in controls]
        else:
            controls = None
        
        out = self.model.diffusion_model(
            x=unet_x,
            timesteps=timesteps,
            context=context,
            y=y,
            time_context=time_context,
            control=controls,
            num_video_frames=num_video_frames,
            image_only_indicator=image_only_indicator,
        )

        return out
    '''

    def configure_optimizers(self):
        lr = self.learning_rate
        params = self.control_model.get_trainable_parameters()
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())

        label_emb_params = []
        for p in self.model.diffusion_model.label_emb.parameters():
            if p.requires_grad: label_emb_params += [p]
        print('\n # label_emb params: ', len(label_emb_params))
        params += label_emb_params

        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())

        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    def on_save_checkpoint(self, checkpoint):
        blacklist = self.control_model.get_blacklist()
        for i in range(len(blacklist)):
            blacklist[i] = 'control_model.' + blacklist[i]

        keys = list(checkpoint['state_dict'].keys() )
        for k in keys:
            names = k.split('.')
            keep_flag = False
            if names[0] == 'control_model': keep_flag = True
            if names[0] == 'model' and names[1] == 'diffusion_model' and names[2] == 'label_emb': keep_flag = True
            if k in blacklist: keep_flag = False
            if not keep_flag:
                del checkpoint['state_dict'][k]

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):

        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.apply_model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)

        # samples = samples * mask + x * (1. - mask)
        return samples

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 14,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        torch.cuda.empty_cache()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
            additional_cond_keys=self.loss_fn.additional_cond_keys,
        )

        sampling_kwargs = {
            key: batch[key] for key in self.loss_fn.batch2model_keys.intersection(batch)
        }

        N = x.shape[0]
        x = x.to(self.device)
        log["inputs"] = x

        torch.cuda.empty_cache()

        z = self.encode_first_stage(x)
        # log["reconstructions"] = self.decode_first_stage(z)
        # log.update(self.log_conditionings(batch, N))

        mask = batch['masks']
        masked_x = x * (1. - mask)
        masked_z = self.encode_first_stage(masked_x)
        mask = F.interpolate(mask, (z.shape[2], z.shape[3]), mode='bilinear', align_corners=False)

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    z, c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            # samples = samples * mask + z * (1. - mask)

            samples = self.decode_first_stage(samples)
            # samples = samples * mask + x * (1. - mask)
            log["samples"] = samples

        return log

    @rank_zero_only
    def log_local(
            self,
            save_dir,
            split,
            images,
            global_step,
            current_epoch,
            batch_idx,
    ):
        root = os.path.join(save_dir, "log_img", split)
        for k in images:
            if isheatmap(images[k]):
                fig, ax = plt.subplots()
                ax = ax.matshow(
                    images[k].cpu().numpy(), cmap="hot", interpolation="lanczos"
                )
                plt.colorbar(ax)
                plt.axis("off")

                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, filename)
                plt.savefig(path)
                plt.close()
                # TODO: support wandb
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                img = Image.fromarray(grid)
                img.save(path)


    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
            images = self.log_images(batch, split="val")

        for k in images:
            # N = min(images[k].shape[0], self.max_images)
            N = images[k].shape[0]
            if not isheatmap(images[k]):
                images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().float().cpu()
                if not isheatmap(images[k]):
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

        self.log_local(
            self.logger.save_dir,
            'val',
            images,
            self.global_step,
            self.current_epoch,
            batch_idx,
        )

        torch.cuda.empty_cache()

    @rank_zero_only
    def test_step(self, batch, batch_idx):
        with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
            images = self.log_images(batch, split="test")

        for k in images:
            # N = min(images[k].shape[0], self.max_images)
            N = images[k].shape[0]
            if not isheatmap(images[k]):
                images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().float().cpu()
                if not isheatmap(images[k]):
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

        self.log_local(
            self.logger.save_dir,
            'test',
            images,
            self.global_step,
            self.current_epoch,
            batch_idx,
        )

        torch.cuda.empty_cache()


    @torch.no_grad()
    def test_infer(self, batch, batch_idx, save_dir):
        with torch.cuda.amp.autocast(**gpu_autocast_kwargs):
            images = self.log_images(batch, split="test")

        for k in images:
            # N = min(images[k].shape[0], self.max_images)
            N = images[k].shape[0]
            if not isheatmap(images[k]):
                images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().float().cpu()
                if not isheatmap(images[k]):
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

        self.log_local(
            save_dir,
            'test',
            images,
            self.global_step,
            self.current_epoch,
            batch_idx,
        )

        torch.cuda.empty_cache()


class SVDEngine(DiffusionEngine):
    def __init__(
            self,
            controlnet_config,
            control_model_path=None,
            init_from_unet=False,
            global_average_pooling=False,
            sd_locked=True,
            drop_first_stage_model=False,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sd_locked = sd_locked
        self.control_model = instantiate_from_config(controlnet_config)
        self.control_scales = [1.0] * 13
        self.global_average_pooling = global_average_pooling

        self.conditioner.embedders[0].eval()
        self.conditioner.embedders[0].requires_grad_(False)
        self.conditioner.embedders[1].eval()
        self.conditioner.embedders[1].requires_grad_(False)
        self.conditioner.embedders[2].eval()
        self.conditioner.embedders[2].requires_grad_(False)
        self.conditioner.embedders[3].eval()
        self.conditioner.embedders[3].requires_grad_(False)
        self.conditioner.embedders[4].eval()
        self.conditioner.embedders[4].requires_grad_(False)

        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)

        if self.sd_locked:
            self.model.diffusion_model.eval()
            self.model.diffusion_model.requires_grad_(False)

        self.control_model.train()
        self.control_model.set_parameters_requires_grad()

        if self.test_ckpt is not None:
            print('## Testing, reload controlnet from test_ckpt!')
            self.control_model.init_ctrl_from_test(self.test_ckpt)
        else:
            if control_model_path is not None:
                self.control_model.init_from_ckpt(control_model_path)
            if init_from_unet:
                missing, unexpected = self.control_model.load_state_dict(self.model.diffusion_model.state_dict(),
                                                                         strict=False)
                print(f"Restored from UNet {len(missing)} missing and {len(unexpected)} unexpected keys")

        if drop_first_stage_model:
            del self.first_stage_model

    def forward(self, x, batch):
        loss = self.loss_fn(self.apply_model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)

        if x.shape[1] == 3:
            x = self.encode_first_stage(x)

        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def apply_model(
            self,
            x: th.Tensor,
            timesteps: th.Tensor,
            cond: Dict,
            time_context: Optional[th.Tensor] = None,
            num_video_frames: Optional[int] = None,
            image_only_indicator: Optional[th.Tensor] = None,

    ):

        B, _, _, _ = x.shape


        cond_concat = cond.get("concat", torch.Tensor([]).type_as(x))
        if 'concat_scale' in cond.keys():
            cond_concat = cond_concat * cond['concat_scale']
        # np.save('vis/concat.npy', cond_concat.detach().cpu().numpy())
        # print('\n# cond_concat: ', cond_concat.size())

        # input_x = torch.cat([x, mask, masked_z, cond_concat], dim=1)
        # input_x_control = torch.cat([x, cond_concat], dim=1)
        input_x = torch.cat([x, cond_concat.type_as(x)], dim=1)
        input_x_control = torch.cat([x, cond_concat.type_as(x)], dim=1)

        context = cond.get('crossattn', None)
        if 'crossattn_scale' in cond.keys():
            context = context * cond['crossattn_scale']
        # np.save('vis/context.npy', context.detach().cpu().numpy())
        # print('\n# context: ', context.size())

        y = cond.get('vector', None)
        # print('\n# y: ', y.size())

        control_hint = cond.get('control_hint', None)
        if 'palette' in cond.keys():
            control_hint = [control_hint, cond['palette']]

        if control_hint is not None:
            controls = self.control_model(
                x=input_x_control,
                hint=control_hint,
                timesteps=timesteps,
                context=context,
                y=y,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            controls = [c * scale for c, scale in zip(controls, self.control_scales)]
            if self.global_average_pooling:
                controls = [torch.mean(c, dim=(2, 3), keepdim=True) for c in controls]
        else:
            controls = None

        out = self.model.diffusion_model(
            x=input_x,
            timesteps=timesteps,
            context=context,
            y=y,
            time_context=time_context,
            control=controls,
            num_video_frames=num_video_frames,
            image_only_indicator=image_only_indicator,
        )

        return out

    '''
    def apply_model_double(
        self,
        cnet_x: th.Tensor,
        unet_x: th.Tensor,
        timesteps: th.Tensor,
        cond: Dict,
        time_context: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,

    ):
        B, _, _, _ = unet_x.shape

        cond_concat = cond.get("concat", torch.Tensor([]).type_as(unet_x))
        if 'concat_scale' in cond.keys():
            cond_concat = cond_concat * cond['concat_scale']
        unet_x = torch.cat([unet_x, cond_concat.type_as(unet_x)], dim=1)
        cnet_x = torch.cat([cnet_x, cond_concat.type_as(cnet_x)], dim=1)

        context = cond.get('crossattn', None)
        if 'crossattn_scale' in cond.keys():
            context = context * cond['crossattn_scale']

        y = cond.get('vector', None)

        control_hint = cond.get('control_hint', None)
        if 'palette' in cond.keys():
            control_hint = [control_hint, cond['palette']]

        if control_hint is not None:
            controls = self.control_model(
                x=cnet_x,
                hint=control_hint,
                timesteps=timesteps,
                context=context,
                y=y,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            controls = [c * scale for c, scale in zip(controls, self.control_scales)]
            if self.global_average_pooling:
                controls = [torch.mean(c, dim=(2, 3), keepdim=True) for c in controls]
        else:
            controls = None

        out = self.model.diffusion_model(
            x=unet_x,
            timesteps=timesteps,
            context=context,
            y=y,
            time_context=time_context,
            control=controls,
            num_video_frames=num_video_frames,
            image_only_indicator=image_only_indicator,
        )

        return out
    '''

    def configure_optimizers(self):
        lr = self.learning_rate
        params = self.control_model.get_trainable_parameters()
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())

        for embedder in self.conditioner.embedders:
            embedder_params = list(embedder.parameters())
            print('\n # Params in Embedders: ', len(embedder_params))
            if embedder.is_trainable:
                print('\n # Set Trainable! ')
                params = params + embedder_params

        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    def on_save_checkpoint(self, checkpoint):
        blacklist = self.control_model.get_blacklist()
        for i in range(len(blacklist)):
            blacklist[i] = 'control_model.' + blacklist[i]

        keys = list(checkpoint['state_dict'].keys())
        for k in keys:
            names = k.split('.')
            keep_flag = False
            if names[0] == 'control_model': keep_flag = True
            if names[0] == 'model' and names[1] == 'diffusion_model' and names[2] == 'label_emb': keep_flag = True
            if k in blacklist: keep_flag = False
            if not keep_flag:
                del checkpoint['state_dict'][k]

    @torch.no_grad()
    def sample(
            self,
            x: torch.Tensor,
            cond: Dict,
            uc: Union[Dict, None] = None,
            batch_size: int = 16,
            shape: Union[None, Tuple, List] = None,
            **kwargs,
    ):

        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.apply_model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)

        # samples = samples * mask + x * (1. - mask)
        return samples

    @torch.no_grad()
    def log_images(
            self,
            batch: Dict,
            N: int = 14,
            sample: bool = True,
            ucg_keys: List[str] = None,
            **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        torch.cuda.empty_cache()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
            additional_cond_keys=self.loss_fn.additional_cond_keys,
        )

        sampling_kwargs = {
            key: batch[key] for key in self.loss_fn.batch2model_keys.intersection(batch)
        }

        N = x.shape[0]
        x = x.to(self.device)
        log["inputs"] = x

        torch.cuda.empty_cache()

        z = self.encode_first_stage(x)
        # log["reconstructions"] = self.decode_first_stage(z)
        # log.update(self.log_conditionings(batch, N))

        mask = batch['masks']
        masked_x = x * (1. - mask)
        masked_z = self.encode_first_stage(masked_x)
        mask = F.interpolate(mask, (z.shape[2], z.shape[3]), mode='bilinear', align_corners=False)

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    z, c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            # samples = samples * mask + z * (1. - mask)

            samples = self.decode_first_stage(samples)
            # samples = samples * mask + x * (1. - mask)
            log["samples"] = samples

        return log

    @rank_zero_only
    def log_local(
            self,
            save_dir,
            split,
            images,
            global_step,
            current_epoch,
            batch_idx,
    ):
        root = os.path.join(save_dir, "log_img", split)
        for k in images:
            if isheatmap(images[k]):
                fig, ax = plt.subplots()
                ax = ax.matshow(
                    images[k].cpu().numpy(), cmap="hot", interpolation="lanczos"
                )
                plt.colorbar(ax)
                plt.axis("off")

                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, filename)
                plt.savefig(path)
                plt.close()
                # TODO: support wandb
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                img = Image.fromarray(grid)
                img.save(path)

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
            images = self.log_images(batch, split="val")

        for k in images:
            # N = min(images[k].shape[0], self.max_images)
            N = images[k].shape[0]
            if not isheatmap(images[k]):
                images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().float().cpu()
                if not isheatmap(images[k]):
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

        self.log_local(
            self.logger.save_dir,
            'val',
            images,
            self.global_step,
            self.current_epoch,
            batch_idx,
        )

        torch.cuda.empty_cache()

    @rank_zero_only
    def test_step(self, batch, batch_idx):
        with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
            images = self.log_images(batch, split="test")

        for k in images:
            # N = min(images[k].shape[0], self.max_images)
            N = images[k].shape[0]
            if not isheatmap(images[k]):
                images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().float().cpu()
                if not isheatmap(images[k]):
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

        self.log_local(
            self.logger.save_dir,
            'test',
            images,
            self.global_step,
            self.current_epoch,
            batch_idx,
        )

        torch.cuda.empty_cache()

    @torch.no_grad()
    def test_infer(self, batch, batch_idx, save_dir):
        with torch.cuda.amp.autocast(**gpu_autocast_kwargs):
            images = self.log_images(batch, split="test")

        for k in images:
            # N = min(images[k].shape[0], self.max_images)
            N = images[k].shape[0]
            if not isheatmap(images[k]):
                images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().float().cpu()
                if not isheatmap(images[k]):
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

        self.log_local(
            save_dir,
            'test',
            images,
            self.global_step,
            self.current_epoch,
            batch_idx,
        )

        torch.cuda.empty_cache()


# w/ blending

class VideoDiffusionEngine2(DiffusionEngine):
    def __init__(
            self,
            controlnet_config,
            control_model_path=None,
            init_from_unet=False,
            global_average_pooling=False,
            sd_locked=True,
            drop_first_stage_model=False,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sd_locked = sd_locked
        self.control_model = instantiate_from_config(controlnet_config)
        self.control_scales = [1.0] * 13
        self.global_average_pooling = global_average_pooling

        self.conditioner.embedders[0].eval()
        self.conditioner.embedders[0].requires_grad_(False)
        self.conditioner.embedders[1].eval()
        self.conditioner.embedders[1].requires_grad_(False)
        self.conditioner.embedders[2].eval()
        self.conditioner.embedders[2].requires_grad_(False)
        self.conditioner.embedders[3].eval()
        self.conditioner.embedders[3].requires_grad_(False)
        self.conditioner.embedders[4].eval()
        self.conditioner.embedders[4].requires_grad_(False)

        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)

        if self.sd_locked:
            self.model.diffusion_model.eval()
            self.model.diffusion_model.requires_grad_(False)

        # label_emb shall be true
        self.model.diffusion_model.label_emb.train()
        self.model.diffusion_model.label_emb.requires_grad_(True)

        self.control_model.train()
        self.control_model.set_parameters_requires_grad()

        if self.test_ckpt is not None:
            print('## Testing, reload controlnet from test_ckpt!')
            self.control_model.init_ctrl_from_test(self.test_ckpt)
        else:
            if control_model_path is not None:
                self.control_model.init_from_ckpt(control_model_path)
            if init_from_unet:
                missing, unexpected = self.control_model.load_state_dict(self.model.diffusion_model.state_dict(),
                                                                         strict=False)
                print(f"Restored from UNet {len(missing)} missing and {len(unexpected)} unexpected keys")

        if drop_first_stage_model:
            del self.first_stage_model

    def forward(self, x, batch):
        loss = self.loss_fn(self.apply_model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)

        if x.shape[1] == 3:
            x = self.encode_first_stage(x)

        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def apply_model(
            self,
            x: th.Tensor,
            timesteps: th.Tensor,
            cond: Dict,
            time_context: Optional[th.Tensor] = None,
            num_video_frames: Optional[int] = None,
            image_only_indicator: Optional[th.Tensor] = None,

    ):

        B, _, _, _ = x.shape

        cond_concat = cond.get("concat", torch.Tensor([]).type_as(x))
        if 'concat_scale' in cond.keys():
            cond_concat = cond_concat * cond['concat_scale']
        # input_x = torch.cat([x, mask, masked_z, cond_concat], dim=1)
        # input_x_control = torch.cat([x, cond_concat], dim=1)
        input_x = torch.cat([x, cond_concat.type_as(x)], dim=1)
        input_x_control = torch.cat([x, cond_concat.type_as(x)], dim=1)

        context = cond.get('crossattn', None)
        if 'crossattn_scale' in cond.keys():
            context = context * cond['crossattn_scale']

        y = cond.get('vector', None)

        control_hint = cond.get('control_hint', None)
        if 'palette' in cond.keys():
            control_hint = [control_hint, cond['palette']]

        if control_hint is not None:
            controls = self.control_model(
                x=input_x_control,
                hint=control_hint,
                timesteps=timesteps,
                context=context,
                y=y,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            controls = [c * scale for c, scale in zip(controls, self.control_scales)]
            if self.global_average_pooling:
                controls = [torch.mean(c, dim=(2, 3), keepdim=True) for c in controls]
        else:
            controls = None

        out = self.model.diffusion_model(
            x=input_x,
            timesteps=timesteps,
            context=context,
            y=y,
            time_context=time_context,
            control=controls,
            num_video_frames=num_video_frames,
            image_only_indicator=image_only_indicator,
        )

        return out

    '''
    def apply_model_double(
        self,
        cnet_x: th.Tensor,
        unet_x: th.Tensor,
        timesteps: th.Tensor,
        cond: Dict,
        time_context: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[th.Tensor] = None,

    ):
        B, _, _, _ = unet_x.shape

        cond_concat = cond.get("concat", torch.Tensor([]).type_as(unet_x))
        if 'concat_scale' in cond.keys():
            cond_concat = cond_concat * cond['concat_scale']
        unet_x = torch.cat([unet_x, cond_concat.type_as(unet_x)], dim=1)
        cnet_x = torch.cat([cnet_x, cond_concat.type_as(cnet_x)], dim=1)

        context = cond.get('crossattn', None)
        if 'crossattn_scale' in cond.keys():
            context = context * cond['crossattn_scale']

        y = cond.get('vector', None)

        control_hint = cond.get('control_hint', None)
        if 'palette' in cond.keys():
            control_hint = [control_hint, cond['palette']]

        if control_hint is not None:
            controls = self.control_model(
                x=cnet_x,
                hint=control_hint,
                timesteps=timesteps,
                context=context,
                y=y,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            controls = [c * scale for c, scale in zip(controls, self.control_scales)]
            if self.global_average_pooling:
                controls = [torch.mean(c, dim=(2, 3), keepdim=True) for c in controls]
        else:
            controls = None

        out = self.model.diffusion_model(
            x=unet_x,
            timesteps=timesteps,
            context=context,
            y=y,
            time_context=time_context,
            control=controls,
            num_video_frames=num_video_frames,
            image_only_indicator=image_only_indicator,
        )

        return out
    '''

    def configure_optimizers(self):
        lr = self.learning_rate
        params = self.control_model.get_trainable_parameters()
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())

        label_emb_params = []
        for p in self.model.diffusion_model.label_emb.parameters():
            if p.requires_grad: label_emb_params += [p]
        print('\n # label_emb params: ', len(label_emb_params))
        params += label_emb_params

        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())

        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    def on_save_checkpoint(self, checkpoint):
        blacklist = self.control_model.get_blacklist()
        for i in range(len(blacklist)):
            blacklist[i] = 'control_model.' + blacklist[i]

        keys = list(checkpoint['state_dict'].keys())
        for k in keys:
            names = k.split('.')
            keep_flag = False
            if names[0] == 'control_model': keep_flag = True
            if names[0] == 'model' and names[1] == 'diffusion_model' and names[2] == 'label_emb': keep_flag = True
            if k in blacklist: keep_flag = False
            if not keep_flag:
                del checkpoint['state_dict'][k]

    @torch.no_grad()
    def sample(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            masked_z: torch.Tensor,
            cond: Dict,
            uc: Union[Dict, None] = None,
            batch_size: int = 16,
            shape: Union[None, Tuple, List] = None,
            **kwargs,
    ):

        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.apply_model, input, sigma, c, **kwargs
        )
        inv_denoiser = lambda input, sigma, c: self.denoiser.inv_sample(
            self.apply_model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, inv_denoiser, x, mask, masked_z, randn, cond, uc=uc)

        # samples = samples * mask + x * (1. - mask)
        return samples

    @torch.no_grad()
    def log_images(
            self,
            batch: Dict,
            N: int = 14,
            sample: bool = True,
            ucg_keys: List[str] = None,
            **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        torch.cuda.empty_cache()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
            additional_cond_keys=self.loss_fn.additional_cond_keys,
        )

        sampling_kwargs = {
            key: batch[key] for key in self.loss_fn.batch2model_keys.intersection(batch)
        }

        N = x.shape[0]
        x = x.to(self.device)
        log["inputs"] = x

        torch.cuda.empty_cache()

        z = self.encode_first_stage(x)
        # log["reconstructions"] = self.decode_first_stage(z)
        # log.update(self.log_conditionings(batch, N))

        mask = batch['masks']
        masked_x = x * (1. - mask)
        masked_z = self.encode_first_stage(masked_x)
        mask = F.interpolate(mask, (z.shape[2], z.shape[3]), mode='bilinear', align_corners=False)

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    z, mask, masked_z, c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = samples * mask + z * (1. - mask)

            clone_x = samples.clone()
            clone_x = clone_x.detach().cpu().numpy()
            np.save(f'logs/demo_out/blended.npy', clone_x)

            samples = self.decode_first_stage(samples)
            # samples = samples * mask + x * (1. - mask)
            log["samples"] = samples

        return log

    @rank_zero_only
    def log_local(
            self,
            save_dir,
            split,
            images,
            global_step,
            current_epoch,
            batch_idx,
    ):
        root = os.path.join(save_dir, "log_img", split)
        for k in images:
            if isheatmap(images[k]):
                fig, ax = plt.subplots()
                ax = ax.matshow(
                    images[k].cpu().numpy(), cmap="hot", interpolation="lanczos"
                )
                plt.colorbar(ax)
                plt.axis("off")

                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, filename)
                plt.savefig(path)
                plt.close()
                # TODO: support wandb
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                img = Image.fromarray(grid)
                img.save(path)

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
            images = self.log_images(batch, split="val")

        for k in images:
            # N = min(images[k].shape[0], self.max_images)
            N = images[k].shape[0]
            if not isheatmap(images[k]):
                images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().float().cpu()
                if not isheatmap(images[k]):
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

        self.log_local(
            self.logger.save_dir,
            'val',
            images,
            self.global_step,
            self.current_epoch,
            batch_idx,
        )

        torch.cuda.empty_cache()

    @rank_zero_only
    def test_step(self, batch, batch_idx):
        with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
            images = self.log_images(batch, split="test")

        for k in images:
            # N = min(images[k].shape[0], self.max_images)
            N = images[k].shape[0]
            if not isheatmap(images[k]):
                images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().float().cpu()
                if not isheatmap(images[k]):
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

        self.log_local(
            self.logger.save_dir,
            'test',
            images,
            self.global_step,
            self.current_epoch,
            batch_idx,
        )

        torch.cuda.empty_cache()

    @torch.no_grad()
    def test_infer(self, batch, batch_idx, save_dir):
        with torch.cuda.amp.autocast(**gpu_autocast_kwargs):
            images = self.log_images(batch, split="test")

        for k in images:
            # N = min(images[k].shape[0], self.max_images)
            N = images[k].shape[0]
            if not isheatmap(images[k]):
                images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().float().cpu()
                if not isheatmap(images[k]):
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

        self.log_local(
            save_dir,
            'test',
            images,
            self.global_step,
            self.current_epoch,
            batch_idx,
        )

        torch.cuda.empty_cache()