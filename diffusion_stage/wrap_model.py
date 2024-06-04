import copy

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import inspect
import diffusers
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MotionDiffusion(nn.Module):
    def __init__(self, cfg, input_length, num_layers, use_upper=False):
        super(MotionDiffusion, self).__init__()
        self.cfg = cfg
        self.scheduler = diffusers.DDIMScheduler(**cfg.scheduler.get("params", dict()))
        self.latent_dim = 512
        self.cond_encoder = nn.Conv1d(3, 22, 1)
        self.cond_encoder2 = nn.Linear(396, self.latent_dim)
        self.denoiser_upper = Denoiser(input_length, num_layers, self.latent_dim, use_upper=use_upper)

        self.mask_training = cfg.mask_traing
        self.mask_num = cfg.mask_num

    def diffusion_reverse(self, sparse, upper_latent=None):
        device = sparse.device
        bs, seq = sparse.shape[:2]
        cond_inter = self.cond_encoder(sparse.flatten(0, 1))  # (bs*seq, 22, 18)
        cond_inter = cond_inter.reshape(bs, seq, -1)  # (bs, seq, 22*18)
        cond = self.cond_encoder2(cond_inter)  # (bs, seq, 384)

        bs, seq, hidden_dim = cond.shape
        latents = torch.randn((bs, seq // 2, 384)).to(device).float()
        latents = latents * self.cfg.init_noise_sigma
        self.scheduler.set_timesteps(self.cfg.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.scheduler.eta
        # reverse
        for i, t in enumerate(timesteps):
            # with torch.no_grad():
            x0_pred = self.denoiser_upper(latents, t.expand(latents.shape[0], ), cond, upper_latent)
            latents = self.scheduler.step(x0_pred, timesteps[i], latents,
                                          **extra_step_kwargs).prev_sample
        return latents


    def forward(self, latents_upper, sparse, upper_latent=None):
        # latents:(bs, seq*4, 384)
        # sparse:(bs, seq, 3, 18)
        bs, seq = sparse.shape[:2]
        device = sparse.device
        cond_inter = self.cond_encoder(sparse.flatten(0, 1))  # (bs*seq, 22, 18)

        if self.training and self.mask_training:
            cond_inter = cond_inter.reshape(bs, seq, 22, 18)
            for i in range(bs):
                mask_index = torch.randint(0, 22, (self.mask_num,))
                cond_inter[i, :, mask_index] = torch.ones_like(cond_inter[i, :, mask_index]) * 0.01

        cond_inter = cond_inter.reshape(bs, seq, -1)  # (bs, seq, 22*18)
        cond = self.cond_encoder2(cond_inter)  # (bs, seq, 384)

        noise = torch.randn_like(latents_upper).float()
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bs,)).to(device)
        timesteps = timesteps.long()
        noisy_latents_upper = self.scheduler.add_noise(latents_upper.clone(), noise, timesteps)
        ori_upper_pred = self.denoiser_upper(noisy_latents_upper, timesteps, cond, upper_latent)

        return ori_upper_pred


class Denoiser(nn.Module):
    def __init__(self, seq_len, num_layers, latent_dim, use_upper=False):
        super(Denoiser, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.use_upper = use_upper

        self.embed_timestep = TimestepEmbedder(self.latent_dim)
        self.sparse_up_conv = nn.Conv1d(self.seq_len, self.seq_len // 2, 1)

        if self.use_upper:
            self.down_dim_init = nn.Linear(384 * 2, 384)
            self.upper_align_net = nn.Conv1d(self.seq_len // 2, self.seq_len // 2, 1)
        else:
            self.align_net = nn.Conv1d(self.seq_len // 2, self.seq_len // 2, 1)
        self.down_dim = nn.Linear(self.latent_dim + 384, self.latent_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(self.latent_dim, 8, mlp_ratio=4) for _ in range(num_layers)
        ])
        nn.init.normal_(self.embed_timestep.mlp[0].weight, std=0.02)
        nn.init.normal_(self.embed_timestep.mlp[2].weight, std=0.02)

        self.last = nn.Linear(self.latent_dim, 384)

    def forward(self, noisy_latents, timesteps, cond, upper_cond=None):
        # noisy_latents:(bs, seq*4, 512)
        # timesteps:(bs, )
        # cond:(bs, seq, 512)
        bs = cond.shape[0]
        timestep_emb = self.embed_timestep(timesteps)  # (batch, 1, 512)

        cond_up4 = self.sparse_up_conv(cond)  # (bs, 4*seq, 512)

        if self.use_upper:
            # noisy_latents = self.align_net(noisy_latents)
            # upper_cond = self.upper_align_net(upper_cond)
            latent_concat = torch.cat((upper_cond, noisy_latents), dim=-1)
            latent_feat = self.down_dim_init(latent_concat)
            latent_feat = self.upper_align_net(latent_feat)
            input_all = torch.cat((cond_up4, latent_feat), dim=-1)
        else:
            noisy_latents = self.align_net(noisy_latents)
            input_all = torch.cat((cond_up4, noisy_latents), dim=-1)
        input_all_512 = self.down_dim(input_all)  # (bs, seq*4, 512)

        x = input_all_512
        for block in self.blocks:
            x = block(x, timestep_emb)
        x = self.last(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
