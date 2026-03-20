import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from demixing.dataset import collate_trials
from demixing.models import SAE


def get_loss_fn(name='MSE'):
    name = name.lower()
    if name == 'mse':
        return nn.MSELoss()
    elif name == 'cosinesimilarity':
        return nn.CosineSimilarity()
    raise ValueError(f'unknown loss: {name}')


def get_optimizer(model, name='Adam', lr=1e-3):
    if name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f'unknown optimizer: {name}')


def compute_loss(pred, z, data, model, loss_fn, l1_weight=0.0, orth_weight=0.0):
    """reconstruction + L1 sparsity + orthogonality on encoder weights"""
    recon_losses = []
    for i in range(pred.shape[0]):
        m = data['mask'][i]
        recon_losses.append(loss_fn(pred[i, m], data['target'][i, m]))
    recon_loss = torch.stack(recon_losses).mean()

    l1_loss = z.abs().mean()

    W = model.to_latent.weight
    W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    gram = W_norm @ W_norm.T
    off_diag = ~torch.eye(gram.shape[0], dtype=bool, device=gram.device)
    ortho_loss = (gram[off_diag] ** 2).mean()

    return recon_loss + l1_weight * l1_loss + orth_weight * ortho_loss


def _train_epoch(loader, model, loss_fn, optimizer, l1_weight, orth_weight):
    model.train()
    for data in loader:
        if isinstance(model, SAE):
            pred, z = model(data['input'])
        else:
            pred, z, _ = model(data['input'], stimulus=data.get('stimulus'))
        loss = compute_loss(pred, z, data, model, loss_fn, l1_weight, orth_weight)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def _test_epoch(loader, model, loss_fn, l1_weight, orth_weight):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            if isinstance(model, SAE):
                pred, z = model(data['input'])
            else:
                pred, z, _ = model(data['input'], stimulus=data.get('stimulus'))
            total_loss += compute_loss(pred, z, data, model, loss_fn,
                                       l1_weight, orth_weight).item()
    return total_loss / len(loader)


def train(dataset, model, ops):
    """
    train a demixing model (SAE or CausalLFADS) on a SpikeData dataset.
    ops should be DEMIXING_OPTIONS from config.py.
    returns dict with 'train' and 'test' loss lists.
    """
    loss_fn = get_loss_fn(ops['loss'])
    optimizer = get_optimizer(model, ops['optimizer'], ops['lr'])

    n_trials = len(dataset)
    n_test = int(n_trials * ops['test_frac'])
    n_train = n_trials - n_test
    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_set, batch_size=ops['batch_size'],
                              shuffle=False, collate_fn=collate_trials)
    test_loader = DataLoader(test_set, batch_size=ops['batch_size'],
                             shuffle=False, collate_fn=collate_trials)

    losses = dict(train=[], test=[])
    for epoch in range(ops['epochs']):
        train_loss = _train_epoch(train_loader, model, loss_fn, optimizer,
                                  ops['l1_weight'], ops['orth_weight'])
        test_loss = _test_epoch(test_loader, model, loss_fn,
                                ops['l1_weight'], ops['orth_weight'])
        print(f'Epoch {epoch + 1}/{ops["epochs"]} | '
              f'train: {train_loss:.5f} | test: {test_loss:.5f}')
        losses['train'].append(train_loss)
        losses['test'].append(test_loss)

    return losses
