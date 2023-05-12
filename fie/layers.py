# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn
from torch_geometric import utils
from torch_scatter import scatter
from einops import rearrange, repeat

from .fisher_embeddings import gmm_attn
from . import ops
from .utils import spherical_kmeans


class FIELayer(gnn.MessagePassing):
    def __init__(self, input_size, output_size, num_mixtures=8, num_heads=1, residue=True,
                 out_proj='kernel', kernel='exp', out_proj_args=0.5, use_deg=False):
        super().__init__(node_dim=0, aggr='add')#, flow='target_to_source')
        self.input_size = input_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures
        self.num_heads = num_heads
        self.residue = residue
        self.use_deg = use_deg

        self.weight = nn.Parameter(
            torch.Tensor(num_mixtures, num_heads, input_size))
        self.bias = nn.Parameter(
            torch.zeros(num_mixtures, num_heads))

        if out_proj == 'relu':
            self.out_proj = nn.Sequential(
                nn.Linear(num_mixtures * num_heads * input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(True)
            )
        elif out_proj == 'kernel':
            self.out_proj = KernelLayer(
                num_mixtures * num_heads * input_size, output_size, sigma=out_proj_args, kernel=kernel)
        else:
            self.out_proj = None

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for w in self.parameters():
            if w.ndim > 2:
                nn.init.normal_(w.data)
                w.data = F.normalize(w.data, dim=-1)

    @torch.no_grad()
    def normalize_(self):
        self.weight.data.copy_(F.normalize(self.weight.data, dim=-1))

    def forward(self, x, edge_index, edge_attr=None, deg_sqrt=None, before_out_proj=False):
        if deg_sqrt is None and self.use_deg:
            deg_sqrt = utils.degree(edge_index[0], x.shape[0], dtype=x.dtype)
            deg_sqrt = (deg_sqrt + 1).log()
        if isinstance(x, torch.Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        # out: n x d x (p x h)
        if self.residue:
            out = out - rearrange(self.weight, "p h d -> 1 d (p h)")
        out = rearrange(out, "n d p -> n (p d)")

        if before_out_proj:
            return out

        if self.out_proj is not None:
            out = self.out_proj(out)

        if self.use_deg:
            out = out * deg_sqrt.view(-1, 1)

        return out

    def feature_transform(self, x_i, x_j):
        return x_j # or (x_i + x_j) * 0.5

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        x_j = self.feature_transform(x_i, x_j)
        x_j_normalized = F.normalize(x_j, dim=-1)
        alpha_ij = gmm_attn(x_j_normalized, self.weight, self.bias, prior=None, eps=self.input_size)
        # alpha_ij: B x hidden_size x num_heads
        alpha_ij = utils.softmax(alpha_ij, index, ptr, size_i)
        x_j = rearrange(x_j, "n d -> n d 1")
        alpha_ij = rearrange(alpha_ij, "n p h -> n 1 (p h)")

        return x_j * alpha_ij

    def forward_proj(self, x, edge_index, deg_sqrt=None):
        if deg_sqrt is None and self.use_deg:
            deg_sqrt = utils.degree(edge_index[0], x.shape[0], dtype=x.dtype)
            deg_sqrt = (deg_sqrt + 1).log()

        if self.out_proj is not None:
            x = self.out_proj(x)
        if self.use_deg and deg_sqrt is not None:
            out = out * deg_sqrt.view(-1, 1)
        return x

    def sample(self, x, edge_index, n_samples=1000):
        indices = torch.randperm(edge_index.shape[1])[:min(edge_index.shape[1], n_samples)]
        edge_index = edge_index[:, indices]
        x_feat = self.feature_transform(x[edge_index[1]], x[edge_index[0]])
        return x_feat

    def unsup_train(self, x, init=None):
        x = F.normalize(x, dim=-1)
        weight = spherical_kmeans(x, self.num_mixtures)
        self.weight.data[:, 0, :].copy_(weight)

        self.normalize_()
        self._need_lintrans_computed = True


class KernelLayer(nn.Module):
    def __init__(self, input_size, output_size, sigma=0.5, kernel='exp'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.reset_parameters()

        alpha = 1. / sigma / sigma
        self.sigma = sigma
        if kernel == 'exp':
            self.kappa = lambda x: torch.exp(alpha * (x - 1.))
        elif kernel == 'linear':
            self.kappa = lambda x: x

        self._need_lintrans_computed = True
        self.register_buffer("lintrans", torch.Tensor(output_size, output_size))

    @torch.no_grad()
    def reset_parameters(self):
        std = 1. / math.sqrt(self.weight.shape[1])
        nn.init.uniform_(self.weight, -std, std)
        self.normalize_()

    @torch.no_grad()
    def normalize_(self):
        self.weight.data.copy_(F.normalize(self.weight.data, dim=-1))

    def train(self, mode=True):
        super().train(mode)
        self._need_lintrans_computed = True

    def _compute_lintrans(self):
        if not self._need_lintrans_computed:
            return self.lintrans
        lintrans = torch.mm(self.weight, self.weight.T)
        lintrans = self.kappa(lintrans)
        lintrans = ops.matrix_inverse_sqrt(lintrans)

        if not self.training:
            self._need_lintrans_computed = False
            self.lintrans.data.copy_(lintrans.data)
        return lintrans

    def forward(self, input):
        self.normalize_()
        norm = torch.norm(input, dim=-1, keepdim=True)
        output = torch.mm(input, self.weight.T) / norm.clamp_min(1e-12)
        output = self.kappa(output)
        output = output * norm

        lintrans = self._compute_lintrans()
        output = torch.mm(output, lintrans)
        return output

    def sample(self, x, n_samples=1000):
        indices = torch.randperm(x.shape[0])[:min(x.shape[0], n_samples)]
        x = x[indices]
        return x

    def unsup_train(self, x, init=None):
        x = F.normalize(x, dim=-1)
        weight = spherical_kmeans(x, self.output_size)
        print(weight.shape)
        self.weight.data.copy_(weight)

        self.normalize_()
        self._need_lintrans_computed = True


class FIEPooling(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=1, residue=True,
                 out_proj='kernel', out_proj_args=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.residue = residue

        self.weight = nn.Parameter(
            torch.Tensor(hidden_size, num_heads, input_size))
        self.bias = nn.Parameter(
            torch.zeros(hidden_size, num_heads))

        if out_proj == 'relu':
            self.out_proj = nn.Sequential(
                nn.Linear(hidden_size * num_heads * input_size, hidden_size),
                nn.ReLU(True)
            )
        elif out_proj == 'kernel':
            self.out_proj = KernelLayer(
                hidden_size * num_heads * input_size, hidden_size, sigma=out_proj_args)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for w in self.parameters():
            if w.ndim > 2:
                nn.init.normal_(w.data)
                w.data = F.normalize(w.data, dim=-1)

    def forward(self, input, batch):
        input_normalized = F.normalize(input, dim=-1)
        attn = gmm_attn(input_normalized, self.weight, self.bias)
        attn = utils.softmax(attn, batch)
        out = rearrange(input, "n d -> n d 1")
        attn = rearrange(attn, "n p h -> n 1 (p h)")
        out = out * attn
        out = scatter(out, batch, dim=0, reduce='add')
        if self.residue:
            out = out - rearrange(self.weight, "p h d -> 1 d (p h)")
        out = rearrange(out, "n d p -> n (p d)")
        out = self.out_proj(out)
        return out


class DiffusionLayer(gnn.MessagePassing):
    def __init__(self, input_size):
        super().__init__(node_dim=0, aggr='add', flow='target_to_source')
        self.input_size = input_size

    def forward(self, x, edge_index, edge_attr=None):
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        return 0.5 * (out + x)

    def message(self, x_i, x_j, edge_attr, index, ptr):
        return x_j
