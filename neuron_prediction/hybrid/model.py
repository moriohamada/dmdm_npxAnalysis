"""
hybrid poisson model: linear skip + independent sub-networks per interaction

fits:
log_rate = skip(x) + interaction_net_0(x[:, cols0]) + interaction_net_1(x[:, cols1]) + ...
         = wx + b  + interaction_net_0(x[:, cols0]) + interaction_net_1(x[:, cols1]) +...


"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class InteractionNet(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.hidden(x)
        reclin = F.relu(h)
        out = self.output(reclin)
        return out

class HybridModel(nn.Module):

    def __init__(self, input_size, interactions, col_map, units_per_group):
        super().__init__()
        self.skip = nn.Linear(input_size, 1)

        self.subnets = nn.ModuleList()
        self.col_ids = []

        for interaction in interactions:
            cols = []
            for group_name in interaction:
                col_slice, _ = col_map[group_name]
                cols.extend(range(col_slice.start, col_slice.stop))
            self.subnets.append(InteractionNet(len(cols), units_per_group))
            self.col_ids.append(torch.tensor(cols, dtype=torch.long))

    def forward(self, x):

        log_rate = self.skip(x).squeeze(-1) # just linear comp

        for subnet, col_ids in zip(self.subnets, self.col_ids):
            log_rate = log_rate + subnet(x[:, col_ids]).squeeze(-1)

        return log_rate.clamp(min=-20, max=20)

    def forward_skip_only(self, x):
        """skip connection only, no interaction networks"""
        return self.skip(x).squeeze(-1).clamp(max=20)

    def forward_lesion(self, x, lesion_idx):
        """lesion interaction"""
        log_rate = self.skip(x).squeeze(-1)
        for i, (subnet, col_ids) in enumerate(zip(self.subnets, self.col_ids)):
            if i == lesion_idx:
                continue
            log_rate = log_rate + subnet(x[:, col_ids]).squeeze(-1)
        return log_rate.clamp(max=20)

    def forward_lesion_skip(self, x, col_map, lesion_groups):
        """zero out specific predictor groups in the skip connection only"""
        log_rate = self.skip(x).squeeze(-1)
        w = self.skip.weight  # (1, n_inputs)
        b = self.skip.bias    # (1,)
        for group_name in lesion_groups:
            col_slice, _ = col_map[group_name]
            log_rate = log_rate - (x[:, col_slice] @ w[:, col_slice].T).squeeze(-1)
        for subnet, col_ids in zip(self.subnets, self.col_ids):
            log_rate = log_rate + subnet(x[:, col_ids]).squeeze(-1)
        return log_rate.clamp(max=20)

def proximal_group_lasso_hybrid(model: HybridModel, col_map, lambda_gl, lr):

    if lambda_gl == 0:
        return

    W = model.skip.weight # (1, n_inputs)
    # linear part
    with torch.no_grad():
        for _, (col_slice,_) in col_map.items():
            group_w = W[:, col_slice]
            group_l2 = group_w.norm(2, dim=1, keepdim=True).clamp(1e-10)
            thresh = lr * lambda_gl * math.sqrt(group_w.shape[1])
            shrink = (1 - (thresh/group_l2)).clamp(min=0)
            W[:, col_slice] = shrink * group_w

    # subnet input layers
    for subnet, interaction in zip(model.subnets, model.interactions):
        W_sub = subnet.hidden.weight  # (n_hidden, n_sub_inputs)
        offset = 0
        with torch.no_grad():
            for group_name in interaction:
                col_slice, _ = col_map[group_name]
                n_cols = col_slice.stop - col_slice.start
                group_w = W_sub[:, offset:offset + n_cols]
                group_l2 = group_w.norm(2, dim=1, keepdim=True).clamp(1e-12)
                thresh = lr * lambda_gl * math.sqrt(n_cols)
                shrink = (1 - (thresh / group_l2)).clamp(min=0)
                W_sub[:, offset:offset + n_cols] = shrink * group_w
                offset += n_cols




