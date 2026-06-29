import math

import torch
import torch.nn as nn


def select_activation(activation: str | None) -> nn.Module:
    if activation == 'elu':
        return nn.ELU()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError('the non_linear_function is not implemented')

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1.):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return torch.nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=0.2, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def anneal_gumbel_temperature(
    epoch: int,
    temp_start: float | None,
    temp_end: float,
    anneal_epochs: int,
    schedule: str = 'cosine',
) -> float:
    """Return Gumbel-Softmax temperature for a 1-indexed training epoch.

    Anneals from ``temp_start`` (exploration, softer) to ``temp_end`` (sharper).
    When ``temp_start`` is None or ``temp_start <= temp_end``, returns constant ``temp_end``.
    """
    if temp_start is None or temp_start <= temp_end or anneal_epochs <= 0:
        return temp_end

    epoch = max(1, epoch)
    if epoch >= anneal_epochs:
        return temp_end
    if anneal_epochs == 1:
        return temp_end

    progress = (epoch - 1) / (anneal_epochs - 1)
    if schedule == 'linear':
        return temp_start + (temp_end - temp_start) * progress
    if schedule == 'cosine':
        return temp_end + (temp_start - temp_end) * (1 + math.cos(math.pi * progress)) / 2
    if schedule == 'exp':
        return temp_start * ((temp_end / temp_start) ** progress)
    raise ValueError(
        f"Unsupported gumbel temperature schedule: {schedule!r}. "
        "Choose from 'linear', 'cosine', 'exp'."
    )


def gumbel_sigmoid(logits, tau: float = 1, hard: bool = False, threshold: float = 0.5):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret