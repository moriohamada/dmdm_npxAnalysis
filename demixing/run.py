"""
per-session demixing: train SAE/LFADS, save model weights + latents
"""
import gc
import torch
import numpy as np
from pathlib import Path

from config import DEMIXING_OPTIONS
from data.session import Session
from utils.filing import get_response_files, load_fr_matrix
from demixing.dataset import SpikeData
from demixing.models import SAE, CausalLFADS
from demixing.train import train
from demixing.analysis import extract_latents, save_latents


def _build_model(n_neurons, ops):
    latent_dim = ops['latent_dim']
    if ops['model_type'] == 'lfads':
        return CausalLFADS(n_neurons=n_neurons, latent_dim=latent_dim,
                           rnn_dim=ops['rnn_dim'])
    return SAE(n_neurons=n_neurons, latent_dim=latent_dim)


def demixing_single_session(sess_dir, ops=DEMIXING_OPTIONS, overwrite=False):
    """train demixing model for one session, save model + latents"""
    sess_dir = Path(sess_dir)
    model_type = ops['model_type']

    latent_path = sess_dir / f'demixing_{model_type}_latents.h5'
    if not overwrite and latent_path.exists():
        return

    sess_pkl = sess_dir / 'session.pkl'
    fr_path = sess_dir / 'FR_matrix.parquet'
    if not sess_pkl.exists() or not fr_path.exists():
        return

    session = Session.load(str(sess_pkl))
    print(f'{session.animal}/{session.name}')

    fr = load_fr_matrix(str(fr_path))
    dataset = SpikeData(session, fr)
    del fr; gc.collect()

    n_neurons = dataset.X.shape[0]
    model = _build_model(n_neurons, ops)

    losses = train(dataset, model, ops)

    z_all = extract_latents(dataset, model)

    # save model weights
    model_path = sess_dir / f'demixing_{model_type}.pt'
    torch.save({
        'state_dict': model.state_dict(),
        'n_neurons': n_neurons,
        'latent_dim': ops['latent_dim'],
        'model_type': model_type,
        'train_loss': losses['train'][-1],
        'test_loss': losses['test'][-1],
    }, str(model_path))

    # save latents
    save_latents(z_all, dataset, model_type, str(latent_path))

    print(f'  saved to {sess_dir.name}')
    del dataset, model, z_all; gc.collect()


def run_demixing(npx_dir, ops=DEMIXING_OPTIONS, n_workers=1, overwrite=False):
    """train demixing model for all sessions"""
    sess_dirs = [str(Path(p).parent) for p in get_response_files(npx_dir)]

    if n_workers <= 1:
        for sess_dir in sess_dirs:
            demixing_single_session(sess_dir, ops, overwrite)
    else:
        from multiprocessing import Pool
        from functools import partial
        with Pool(n_workers, maxtasksperchild=1) as pool:
            pool.map(partial(demixing_single_session, ops=ops,
                             overwrite=overwrite), sess_dirs)
