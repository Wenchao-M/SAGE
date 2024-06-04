# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import math
import torch
from einops import rearrange
from torch import nn
from VQVAE.blocks.mingpt import Block
from VQVAE.blocks.convolutions import Masked_conv, Masked_up_conv


class PositionalEncoding(nn.Module):
    def __init__(self, dim, type='sine_frozen', max_len=1024, *args, **kwargs):
        super(PositionalEncoding, self).__init__()
        if 'sine' in type:
            rest = dim % 2
            pe = torch.zeros(max_len, dim + rest)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim + rest, 2).float() * (-math.log(10000.0) / (dim + rest)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe[:, :dim]
            pe = pe.unsqueeze(0)  # [1,t,d]
            if 'ft' in type:
                self.pe = nn.Parameter(pe)
            elif 'frozen' in type:
                self.register_buffer('pe', pe)
            else:
                raise NameError
        elif type == 'learned':
            self.pe = nn.Parameter(torch.randn(1, max_len, dim))
        elif type == 'none':
            # no positional encoding
            pe = torch.zeros((1, max_len, dim))  # [1,t,d]
            self.register_buffer('pe', pe)
        else:
            raise NameError

    def forward(self, x, start=0):
        x = x + self.pe[:, start:(start + x.size(1))]
        return x


class Encoder_Config:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    n_embd = 384

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class Stack(nn.Module):
    """ A stack of transformer blocks.
        Used to implement a U-net structure """

    def __init__(self, block_size, n_layer=12, n_head=8, n_embd=256, dropout=0.1, causal=False, down=1, up=1,
                 pos_type='sine_frozen', sample_method='conv', pos_all=False):
        super().__init__()
        config = Encoder_Config(block_size, n_embd=n_embd, n_layer=n_layer,
                                n_head=n_head, dropout=dropout, causal=causal)
        self.drop = nn.Dropout(dropout)
        assert down == 1 or up == 1, "Unexpected combination"
        assert down in [1, 2] and up in [1, 2], "Not implemented"
        assert sample_method in ['cat', 'conv'], "Unknown sampling method"
        cat_down, slice_up = (down, up) if sample_method == 'cat' else (1, 1)
        self.cat_down, self.slice_up = cat_down, slice_up
        self.pos_all = pos_all
        self.blocks = nn.ModuleList([])
        self.pos = nn.ModuleList([])
        for i in range(config.n_layer):
            # Inside Block, standard transformer stuff happens.
            self.blocks.append(Block(config,
                                     in_factor=cat_down if i == 0 and cat_down > 1 else None,
                                     out_factor=slice_up if i == config.n_layer - 1 and slice_up > 1 else None))
            in_dim = config.n_embd * (cat_down if i == 0 and cat_down > 1 else 1)
            if pos_all or i == 0:
                self.pos.append(PositionalEncoding(dim=in_dim, max_len=block_size, type=pos_type))
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        # print("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.down_conv, self.up_conv = None, None
        if sample_method == 'conv':
            if down == 2:
                self.down_conv = Masked_conv(config.n_embd, config.n_embd,
                                             pool_size=down, pool_type='max')
            elif up == 2:
                self.up_conv = Masked_up_conv(config.n_embd, config.n_embd)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x=None, z=None):
        assert (x is None) ^ (z is None), "Only x or z as input"
        x = x if x is not None else z
        t = x.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        if self.cat_down > 1:
            if self.cat_down != 2:
                raise NotImplementedError
            else:
                x = rearrange(x, 'b (t t2) c -> b t (t2 c)', t2=2)

        if self.down_conv is not None:
            x = self.down_conv(x)

        x = self.drop(x)
        for i in range(len(self.blocks)):
            x = self.pos[i](x) if (i == 0 or self.pos_all) else x
            x = self.blocks[i](x, in_residual=not (i == 0 and self.cat_down > 1),
                               out_residual=not (i == (len(self.blocks) - 1) and self.slice_up > 1))
        if self.slice_up > 1:
            x = rearrange(x, 'b t (t2 c) -> b (t t2) c', t2=2)

        if self.up_conv is not None:
            x = self.up_conv(x)

        x = self.ln_f(x)  # (bs, seq/2, 384)
        logits = self.head(x)  # (bs, seq/2, 384)
        return logits


class TransformerDecoder(nn.Module):
    def __init__(self, *, in_dim=84, seq_len=40, n_layers=4, hid_dim=384, heads=4, e_dim=256,
                 block_size=2048, pos_type='sine_frozen', pos_all=False, **kwargs):
        super().__init__()
        self.seq_len = seq_len

        # encoder decoder related config
        n_embd = hid_dim
        self.in_dim = in_dim

        self.cond_encoder = nn.Conv1d(3, 22, 1)
        self.cond_encoder2 = nn.Linear(396, 384)
        self.sparse_up_conv = nn.Conv1d(20, 10, 1)
        self.post_quant_emb = nn.Linear(e_dim * 2 + 384, n_embd)

        # Build the decoder
        self.decoder_stacks = nn.ModuleList([
            Stack(block_size=block_size, n_layer=n_layers, n_head=heads, n_embd=n_embd,
                                                   causal=False, up=2, pos_type=pos_type, pos_all=pos_all)
        ])
        dim = n_embd
        # Final head to predict body and root paramaters
        self.reg_body = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, self.in_dim - 6)
        )
        self.reg_root = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 6)
        )

    def forward(self, upper_latent, lower_latent, sparse):
        bs, seq = sparse.shape[:2]
        bs, latent_seq = upper_latent.shape[:2]
        sparse = sparse.reshape(bs * seq, 3, 18)
        cond = self.cond_encoder(sparse)
        cond = cond.reshape(bs, seq, -1)
        cond = self.cond_encoder2(cond)
        cond = self.sparse_up_conv(cond)
        ul_latent = torch.cat((cond, upper_latent, lower_latent), dim=-1)
        hid = self.post_quant_emb(ul_latent)  # this one is i.i.d
        y = self.decoder_stacks[0](z=hid)
        rotmat, rotroot = self.reg_body(y), self.reg_root(y)
        rotmat = torch.cat((rotroot, rotmat), dim=-1)
        rotmat = rotmat.reshape(bs, -1, 132)
        return rotmat
