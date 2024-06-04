# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from argparse import ArgumentParser
import ipdb

""" VectorQuantizer code adapted from by https://github.com/CompVis/taming-transformers/taming/modules/vqvae/quantize.py"""

__all__ = ['VectorQuantizer']


def L2_efficient(x, y):
    return (x.pow(2).sum(1, keepdim=True) - 2 * x @ y + y.pow(2).sum(0, keepdim=True))


# def cos_efficient(x, y):
#     # x:(bs*seq_num, dim//n_book)   y:(dim//n_book, n_e//n_book)
#     normed_x = F.normalize(x, dim=1)
#     normed_codebook = F.normalize(y, dim=0)
#     res = torch.einsum('bd,dn->bn', normed_x, normed_codebook)
#     return res


class EmaCodebookMeter:
    """Compute an estimate of centroid usage, using an EMA to track proportions """

    def __init__(self, codebook_size, ema_alpha=0.05):
        self.codebook_size = codebook_size
        self.bins = (torch.ones((self.codebook_size), requires_grad=False) / self.codebook_size).detach().cuda()
        self.ema_alpha = ema_alpha
        self.iters = 0

    def bincount(self, val, weights=None):
        norm = val.shape[0]
        weights = weights.reshape(-1) if weights is not None else None
        count = torch.bincount(val.reshape(-1), minlength=self.codebook_size,
                               weights=weights).detach()
        self.iters += 1
        return count / norm

    def load(self, bins):
        self.bins = torch.tensor(bins, requires_grad=False).detach().cuda()

    def update(self, val, weights=None, n=1):
        """ Count usage of each value in the codebook """
        count = self.bincount(val, weights=weights)
        alpha = max(self.ema_alpha, 1 / (self.iters + 1))
        self.bins = (1. - alpha) * self.bins + alpha * count

    def get_hist(self):
        return self.bins


class VectorQuantizer(nn.Module):
    """
    Code taken from https://github.com/CompVis/taming-transformers/
            blob/9d17ea64b820f7633ea6b8823e1f78729447cb57/taming/
            modules/vqvae/quantize.py#L213
    for handling input of shape [batch_size, seq_len, hid_dim]

    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    def __init__(self, n_e, e_dim, beta,
                 nbooks=1, balance=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.nbooks = nbooks
        self.balance = balance

        assert n_e % nbooks == 0, "nb codebooks should divide nb centroids"
        self.n_e_i = n_e // nbooks

        embed_dims = (nbooks - 1) * [e_dim // nbooks] + \
                     [e_dim - (nbooks - 1) * (e_dim // nbooks)]
        self.embed_dims = embed_dims

        self.embeddings = torch.nn.ModuleDict({str(i): nn.Embedding(self.n_e_i, d) for i, d in enumerate(embed_dims)})

        # self.trackers = {}
        for i, e in self.embeddings.items():
            e.weight.data.uniform_(-1.0 / self.n_e_i, 1.0 / self.n_e_i)
        print(f"Codebook {i}: {list(e.weight.size())}")

            # self.trackers[int(i)] = EmaCodebookMeter(self.n_e_i)

        self.decay = 0.99
        self.register_buffer("embed_prob", torch.zeros(self.n_e_i, self.nbooks))
        self.init = False
        self.anchor = 'closest'
        self.first_batch = False
        self.contras_loss = True

    def forward_one(self, z, i):
        bsize = self.e_dim // self.nbooks
        e_dim = bsize if i < self.nbooks - 1 else self.e_dim - (self.nbooks - 1) * bsize

        z_flattened = z.view(-1, e_dim)
        dist = -L2_efficient(z_flattened, self.embeddings[str(i)].weight.t())
        # dist = cos_efficient(z_flattened, self.embeddings[str(i)].weight.t())

        min_encoding_indices = torch.argmax(dist, dim=1)
        min_encodings = torch.nn.functional.one_hot(min_encoding_indices, num_classes=self.n_e_i).float()

        # if self.training:
        #     self.track_assigment(min_encoding_indices.detach(), i)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embeddings[str(i)].weight).view(z.shape)

        # min_encoding_indices.view(z.shape)
        return z_q, min_encoding_indices.view(z.shape[:-1] + (1,)), min_encodings, dist, z_flattened

    def forward(self, z, p=1.0):
        assert z.size(2) == self.e_dim
        zs = torch.split(z, z.size(2) // len(self.embeddings), dim=-1)
        zq_i = [self.forward_one(z, i) for i, z in enumerate(zs)]
        # z_q:(bs, seq/2, 256)   min_encoding_indices:(bs, 16, 2)
        z_q, min_encoding_indices = [torch.cat([e[i] for e in zq_i], dim=-1) for i in [0, 1]]
        min_encodings, d, z_flattened = [[e[i] for e in zq_i] for i in range(2, 5)]

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2, dim=-1) + self.beta * \
               torch.mean((z_q - z.detach()) ** 2, dim=-1)

        if p != 1.0:
            # Proba of being quantized.
            quant_mask = torch.bernoulli(p * torch.ones_like(z)).float()
            z_q = quant_mask * z_q + (1 - quant_mask) * z

        # preserve gradients
        z_q = z + (z_q - z).detach()

        avg_probs = [torch.mean(min_encodings[i], dim=0) for i in range(self.nbooks)]
        if self.training:
            # calculate the average usage of code entries
            for i in range(self.nbooks):
                self.embed_prob[..., i].mul_(self.decay).add_(avg_probs[i], alpha=1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                sort_distance = []
                indices = []
                for i in range(self.nbooks):
                    sort_distance_cur, indices_cur = d[i].sort(dim=0)
                    sort_distance.append(sort_distance_cur)
                    indices.append(indices_cur)
                random_feat = [z_flattened[i].detach()[indices[i][-1, :]] for i in range(self.nbooks)]

                # decay parameter based on the average usage
                decay = [torch.exp(-(self.embed_prob[:, i] * self.n_e * 10) / (1 - self.decay) - 1e-3).unsqueeze(
                    1).repeat(1, self.e_dim // self.nbooks) for i in range(self.nbooks)]
                for i in range(self.nbooks):
                    self.embeddings[f"{i}"].weight.data = self.embeddings[f"{i}"].weight.data * (1 - decay[i]) + \
                                                          random_feat[i] * decay[i]
                if self.first_batch:
                    self.init = True

            # contrastive loss
            if self.contras_loss:
                dis_pos = sort_distance[0][-max(1, sort_distance[0].size(0) // self.n_e):, :].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[0][:int(sort_distance[0].size(0) * 1 / 2), :]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07

                dis_pos2 = sort_distance[1][-max(1, sort_distance[0].size(0) // self.n_e):, :].mean(dim=0, keepdim=True)
                dis_neg2 = sort_distance[1][:int(sort_distance[0].size(0) * 1 / 2), :]
                dis2 = torch.cat([dis_pos2, dis_neg2], dim=0).t() / 0.07

                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                contra_loss2 = F.cross_entropy(dis2, torch.zeros((dis2.size(0),), dtype=torch.long, device=dis.device))
                loss = loss + contra_loss + contra_loss2

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, eos_mask=None):
        """
        Args:
            - indices: [batch_size,seq_len]
        Return:
            - z_q: [batch_size,seq_len,e_dim]
        """
        # This is a hack, but it enables us to keep the '-1' index solely in the gpt
        embds = [self.embeddings[str(i)](e.squeeze(-1)) for i, e in enumerate(torch.split(indices, 1, dim=-1))]
        return torch.cat(embds, dim=-1)
