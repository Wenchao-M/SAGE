import torch
import torch.nn as nn


class Refinenet(nn.Module):
    def __init__(self, n_layers, hidder_dim):
        super().__init__()
        self.n_layers = n_layers # 2
        self.hidden_dim = hidder_dim  # 512
        self.refine_net = nn.GRU(132, self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, 132)

    def forward(self, x, hidden):
        # x:(1, seq, 132)
        bs, seq = x.shape[:2]
        if hidden is None:
            self.init_hidden(bs)
        feat, hidden = self.refine_net(x, hidden)
        out = self.linear(feat)
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
