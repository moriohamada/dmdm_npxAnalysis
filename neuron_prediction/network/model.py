"""poisson models for single-neuron encoding"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PoissonLinear(nn.Module):
    """linear poisson model - same as GLM but trained with SGD + group lasso"""
    def __init__(self, n_inputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)

    def forward(self, x):
        log_rate = self.linear(x).squeeze(-1)
        return log_rate.clamp(max=20)


class PoissonNet(nn.Module):
    """single hidden layer network with exponential (Poisson) output

    predicts log(firing rate) from the same design matrix used by the GLM
    """
    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.output = nn.Linear(n_hidden, 1)

    def forward(self, x):
        h = F.relu(self.hidden(x))
        log_rate = self.output(h).squeeze(-1)
        return log_rate.clamp(max=20)


def proximal_group_lasso(model, col_map, lambda_gl, lr):
    """proximal operator for group lasso (block soft-thresholding)

    applied after each SGD step on the smooth loss.
    for each group g: w_g <- w_g * max(0, 1 - lr * lambda * sqrt(df_g) / ||w_g||)
    this zeros out entire groups when their norm is below the threshold

    operates on first-layer weights:
      PoissonLinear.linear.weight or PoissonNet.hidden.weight
    """
    if lambda_gl <= 0:
        return

    if isinstance(model, PoissonLinear):
        W = model.linear.weight  # (1, n_inputs)
    else:
        W = model.hidden.weight  # (n_hidden, n_inputs)

    with torch.no_grad():
        for name, (col_slice, _) in col_map.items():
            group_w = W[:, col_slice]  # (n_units, n_cols_in_group)
            df_g = group_w.shape[1]
            threshold = lr * lambda_gl * math.sqrt(df_g)

            # per-unit L2 norm of this group's weights
            norms = group_w.norm(dim=1, keepdim=True).clamp(min=1e-12)
            scale = (1 - threshold / norms).clamp(min=0)
            W[:, col_slice] = group_w * scale


def ortho_penalty(model):
    """mean squared off-diagonal cosine similarity of hidden unit weight vectors

    encourages functionally distinct hidden units.
    uses mean (not sum) so penalty strength is stable across hidden widths.
    only meaningful for PoissonNet (not PoissonLinear)
    """
    if isinstance(model, PoissonLinear):
        return torch.tensor(0.0)
    W = model.hidden.weight  # (n_hidden, n_inputs)
    W_norm = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos_sim = W_norm @ W_norm.T
    mask = 1 - torch.eye(cos_sim.shape[0], device=cos_sim.device)
    return (cos_sim * mask).pow(2).mean()
