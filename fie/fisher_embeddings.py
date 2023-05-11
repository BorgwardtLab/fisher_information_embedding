import math
import torch
import torch.nn.functional as F
from torch_geometric import utils
from einops import rearrange


def gmm_attn(x, mu, log_pi=0, prior=None, eps=1.0, gamma=1.0):
    """Fisher information embedding using Gaussian mixture
    Input:
    x_i: n x d
    mu_j: p x h x d
    log_pi_j: p x h bias term
    Output:
    alpha_ij: n x p x h
    """
    p, h, d = mu.shape
    mu = rearrange(mu, "p h d -> (p h) d")
    log_pi = rearrange(log_pi, "p h -> (p h)")
    attn = F.linear(x, mu, bias=log_pi)
    # attn: n x (p x h)
    attn = rearrange(attn, "n (p h) -> n p h", p=p)
    if prior is None:
        attn = F.softmax(attn / eps, dim=1)
    elif prior == 'ot':
        attn = rearrange(attn, "n p h -> (n h) p")
        attn = ot(attn, eps=eps)
        attn = rearrange(attn, "(n h) p -> n p h", h=h)
    elif prior == 'uot':
        attn = rearrange(attn, "n p h -> (n h) p")
        attn = uot(attn, eps=eps, gamma=gamma)
        attn = rearrange(attn, "(n h) p -> n p h", h=h)
    return attn


def ot(K, a=None, b=None, eps=1.0, max_iter=5):
    m, n = K.shape
    v = K.new_zeros((m,))
    if a is None:
        a = 0
    else:
        a = torch.log(a)
    if b is None:
        b = math.log(m / n)
    else:
        b = torch.log(b)

    K = K / eps

    for _ in range(max_iter):
        u = -torch.logsumexp(v.view(m, 1) + K, dim=0) + b
        v = -torch.logsumexp(u.view(1, n) + K, dim=1) + a

    return torch.exp(K + u.view(1, n) + v.view(m, 1))


def uot(K, a=None, b=None, eps=1.0, gamma=1.0, max_iter=5):
    m, n = K.shape
    v = K.new_zeros((m,))
    if a is None:
        a = 0
    else:
        a = torch.log(a)
    if b is None:
        b = math.log(m / n)
    else:
        b = torch.log(b)

    fi = gamma / (gamma + eps)
    K = K / eps

    for _ in range(max_iter):
        u = (-torch.logsumexp(v.view(m, 1) + K, dim=0) + b) * fi
        v = -torch.logsumexp(u.view(1, n) + K, dim=1) + a

    return torch.exp(K + u.view(1, n) + v.view(m, 1))
