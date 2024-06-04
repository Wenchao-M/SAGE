# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import math
import logging
import torch
from einops import rearrange
from torch import nn
from .blocks.mingpt import Block
from .blocks.convolutions import Masked_conv, Masked_up_conv
from .quantize import VectorQuantizer

logger = logging.getLogger(__name__)


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

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class Stack(nn.Module):
    """ A stack of transformer blocks.
        Used to implement a U-net structure """

    def __init__(self, block_size, n_layer=12, n_head=8, n_embd=256,
                 dropout=0.1, causal=False, down=1, up=1,
                 pos_type='sine_frozen', sample_method='conv',
                 pos_all=False):
        super().__init__()
        config = Encoder_Config(block_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, dropout=dropout,
                                causal=causal)
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
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.down_conv, self.up_conv = None, None
        if sample_method == 'conv':
            if down == 2:
                self.down_conv = Masked_conv(config.n_embd, config.n_embd, pool_size=down, pool_type='max')
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

    def forward(self, x=None):
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


class TransformerAutoEncoder(nn.Module):
    """
    Model composed of an encoder and a decoder.
    """

    # NOTE This is an abstract class for us
    # as we are not interested in vanilla autoencoders 
    # with low dimensionality bottlenecks, so it does not implement forward().
    def __init__(self, in_dim=84, n_layers=[4, 4], hid_dim=256, heads=4, dropout=0.0, e_dim=384, block_size=2048,
                 pos_type='sine_frozen', pos_all=False, sample_method='conv'):
        super().__init__()

        if not isinstance(hid_dim, list):
            hid_dim = [hid_dim]
        if len(hid_dim) != 1:
            raise NotImplementedError("Does not handle per-layer channel specification.")

        # Constrols masking of  attention in encoder / decoder.
        n_embd = hid_dim[0]
        self.in_dim = in_dim
        num_joints = in_dim // 6
        self.emb = nn.Linear(in_dim + num_joints * 18, n_embd)
        self.up_sparse = nn.Conv1d(3, num_joints, 1)

        # Build the encoder; basic brick is a 'Stack object'.
        self.encoder_stacks = nn.ModuleList(
            [Stack(block_size=block_size, n_layer=n_layers[0], n_head=heads, n_embd=n_embd,
                   dropout=dropout, down=2, pos_type=pos_type,
                   pos_all=pos_all, sample_method=sample_method)])

        # project features (hid) to latent variable dimensions (before going through bottleneck)
        # and then z to hid
        self.emb_in, self.emb_out = n_embd, e_dim
        self.quant_emb = nn.Linear(n_embd, e_dim)
        self.encoder_dim = e_dim
        self.post_quant_emb = nn.Linear(e_dim, n_embd)

        # Build the decoder
        self.decoder_stacks = nn.ModuleList(
            [Stack(block_size=block_size, n_layer=n_layers[1], n_head=heads, n_embd=n_embd,
                   up=2, pos_type=pos_type, pos_all=pos_all)])
        dim = n_embd
        # Final head to predict body and root paramaters
        self.reg_body = nn.Sequential(nn.Linear(dim, dim),
                                      nn.ReLU(), nn.Linear(dim, self.in_dim))

    def encoder(self, x):
        """ Calls each encoder stack sequentially """
        o = self.encoder_stacks[0](x)
        return o

    def decoder(self, z):
        """ Calls each decoder stack sequentially """
        o = self.decoder_stacks[0](z)
        return o

    def regressor(self, x):
        return self.reg_body(x)  # , self.reg_root(x)


class TransformerVQVAE(TransformerAutoEncoder):
    """
    Adds a quantization bottleneck to TransformerAutoEncoder.
    """

    def __init__(self, in_dim=84, n_layers=[4, 4], hid_dim=256, heads=4, dropout=0., n_codebook=8, n_e=512,
                 e_dim=384, beta=1.):
        super().__init__(**{'in_dim': in_dim, 'n_layers': n_layers, 'hid_dim': hid_dim, 'e_dim': e_dim, 'heads': heads,
                            'dropout': dropout})
        assert e_dim % n_codebook == 0
        assert n_e % n_codebook == 0
        self.one_codebook_size = n_e // n_codebook
        assert not any([a is None for a in [n_e, e_dim, beta]]), "Missing arguments"
        self.quantizer = VectorQuantizer(n_e=n_e, e_dim=e_dim, beta=beta, nbooks=n_codebook)

    def forward_encoder(self, x, sparse):
        """"
        Run the forward pass of the encoder
        """
        bs, seq, *_ = x.size()
        sparse_emb = self.up_sparse(sparse.reshape(bs * seq, 3, 18))  # (bs*seq, 22, 18)
        x = x.reshape(bs * seq, -1, 6)
        x_all = torch.cat((x, sparse_emb), dim=-1)
        x_all = x_all.reshape(bs, seq, -1)  # (bs, seq, 22*(18+6))
        x_emb = self.emb(x_all)  # (bs, seq, 512)
        hid = self.encoder(x=x_emb)  # hid:(bs, seq/2, 384)  mask:(bs, seq/2)
        return hid

    def forward_decoder(self, z):
        bs, seq_len, *_ = z.shape
        return self.decoder_stacks[0](z)

    def forward(self, x, sparse, quant_prop=1.0):
        batch_size, seq_len, *_ = x.size()
        hid = self.forward_encoder(x=x, sparse=sparse)  # hid:(bs, seq * k, 512)   mask_:(bs, seq*k)
        z = self.quant_emb(hid)
        z_q, z_loss, indices = self.quantize(z, p=quant_prop)  # z_q:(batch, 16, 512)
        hid = self.post_quant_emb(z_q)  # this one is i.i.d
        y = self.decoder(z=hid)
        rotmat = self.regressor(y)
        rotmat = rotmat.reshape(batch_size, seq_len, -1)
        kl = math.log(self.one_codebook_size) * torch.ones_like(indices, requires_grad=False)
        return rotmat, {'quant_loss': z_loss, 'kl': kl}, indices

    def quantize(self, z, p=1.0):
        z_q, loss, indices = self.quantizer(z, p=p)  # z:(batch, seq/2, 256)
        return z_q, loss, indices

    def encode_my(self, x, sparse, quant_prop=1.0):
        batch_size, seq_len, *_ = x.size()
        hid = self.forward_encoder(x=x, sparse=sparse)  # hid:(bs, seq * k, 512)   mask_:(bs, seq*k)
        z = self.quant_emb(hid)
        z_q, z_loss, indices = self.quantize(z, p=quant_prop)  # z_q:(batch, 16, 512)
        return z_q

    def decode_my(self, z, batch_size, seq_len):
        hid = self.post_quant_emb(z)  # this one is i.i.d
        y = self.forward_decoder(z=hid)
        # rotroot = self.reg_root(y)
        rotmat = self.reg_body(y)
        # rotmat = torch.cat((rotroot, rotmat), dim=-1)
        rotmat = rotmat.reshape(batch_size, seq_len, -1)
        return rotmat
