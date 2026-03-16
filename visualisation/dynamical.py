"""
Visualisation for LDS results: flow fields and projected trajectories.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set_style("whitegrid")

from config import ANALYSIS_OPTIONS, PATHS, PLOT_COLOURS
from data.session import Session
from analyses.load_responses import load_psth_mean
from analyses.dynamical import CONDITIONS, _get_condition_mask, _build_input_vector
from utils.filing import get_response_files


def get_dynamics_plane(A, Z_valid, method='flow'):
    """
    Find 2D basis for visualising dynamics.

    Args:
        A: (n_pcs, n_pcs) dynamics matrix
        Z_valid: (n_pcs, T_valid) PC trajectories (only valid/masked bins)
        method:
            'pc'   — just use PC dimensions 1 and 2
            'flow' — PCA on the flow vectors dx = (A-I)x; finds the plane
                     where state is changing most, which is usually more
                     informative than raw variance
    Returns:
        basis: (n_pcs, 2) orthonormal column vectors spanning the plane
    """
    n_pcs = A.shape[0]
    if method == 'pc':
        basis = np.eye(n_pcs, 2)
    elif method == 'flow':
        flow = (A - np.eye(n_pcs)) @ Z_valid
        U, _, _ = np.linalg.svd(flow, full_matrices=False)
        basis = U[:, :2]
    else:
        raise ValueError(f"Unknown method: {method}")
    return basis


def plot_flow_field(A, basis, ax,
                    grid_range=(-3, 3), n_grid=15,
                    color='k', alpha=0.4):
    """
    Plot 2D quiver flow field for dynamics matrix A, projected onto
    the given 2D basis.

    For each point p on a grid in the 2D plane, computes the flow
    dp = (A-I)x where x is the lift of p back to n_pcs space,
    then projects the flow back to 2D.
    """
    n_pcs = A.shape[0]
    A_eff = A - np.eye(n_pcs)

    # project A into the 2D plane: dp_2d = basis.T @ A_eff @ basis @ p_2d
    A_2d = basis.T @ A_eff @ basis

    g = np.linspace(grid_range[0], grid_range[1], n_grid)
    P1, P2 = np.meshgrid(g, g)
    U = A_2d[0, 0] * P1 + A_2d[0, 1] * P2
    V = A_2d[1, 0] * P1 + A_2d[1, 1] * P2

    ax.quiver(P1, P2, U, V, color=color, alpha=alpha, scale=None)
    ax.set_aspect('equal')


def project_to_plane(mean_psth, weights, basis):
    """
    Project a mean PSTH (nN x nT) through PCA weights and onto 2D plane.
    Returns (2, nT) trajectory in the visualisation plane.
    """
    z = weights.T @ mean_psth   # (n_pcs, nT)
    return basis.T @ z           # (2, nT)


def plot_trajectory(traj_2d, t_ax, ax,
                    color='k', label=None, lw=1.5,
                    marker_start=True):
    """
    Plot a 2D trajectory with optional start marker.
    traj_2d: (2, nT)
    """
    ax.plot(traj_2d[0], traj_2d[1], color=color, lw=lw, label=label)
    if marker_start:
        ax.plot(traj_2d[0, 0], traj_2d[1, 0], 'o', color=color, ms=6)


def _load_weights_and_basis(sess_dir, pca_key, lds_cond, fr_matrix, session,
                            ops, plane_method='flow'):
    """
    Load PCA weights and compute the 2D visualisation plane from
    a fitted LDS condition.
    """
    with h5py.File(sess_dir / 'pca.h5', 'r') as f:
        weights = f[pca_key]['weights'][:]

    with h5py.File(sess_dir / f'lds_{pca_key}.h5', 'r') as f:
        A = f[lds_cond]['A'][:]

    # get valid bins for this condition to compute the flow plane
    t_ax = fr_matrix.columns.values
    Z = weights.T @ fr_matrix.values
    mask = _get_condition_mask(session, t_ax, lds_cond, ops)
    Z_valid = Z[:, mask]

    basis = get_dynamics_plane(A, Z_valid, method=plane_method)
    return weights, A, basis


def plot_baseline_trajectories(psth_path, weights, basis, ax):
    """Baseline onset: early vs late block."""
    c = PLOT_COLOURS['block']
    for block, color in c.items():
        mean, _, t = load_psth_mean(psth_path, 'blOn', block,
                                    baseline_subtract=True)
        traj = project_to_plane(mean, weights, basis)
        plot_trajectory(traj, t, ax, color=color, label=f'{block} block')
    ax.set_title('Baseline onset')
    ax.legend(fontsize=7)


def plot_tf_trajectories(psth_path, weights, basis, ax,
                         block_label='lateBlock_early'):
    """
    TF pulses for one block/time condition, pos vs neg.
    block_label: e.g. 'earlyBlock_early', 'lateBlock_early', 'lateBlock_late'
    """
    c = PLOT_COLOURS['tf']
    for pol, color in c.items():
        cond = f'{block_label}_{pol}'
        mean, _, t = load_psth_mean(psth_path, 'tf', cond,
                                    baseline_subtract=True)
        traj = project_to_plane(mean, weights, basis)
        plot_trajectory(traj, t, ax, color=color, label=pol)
    ax.set_title(f'TF: {block_label}')
    ax.legend(fontsize=7)


def plot_lick_trajectories(psth_path, weights, basis, ax,
                           outcome='fa'):
    """
    Lick-aligned trajectories split by block, for hits or FAs.
    """
    c = PLOT_COLOURS['block']
    line_styles = {
        'earlyBlock_early': '-',
        'lateBlock_early': '-',
        'lateBlock_late': '--',
    }
    for cond_key, ls in line_styles.items():
        block = 'early' if cond_key.startswith('early') else 'late'
        color = c[block]
        cond = f'{cond_key}_{outcome}'
        try:
            mean, _, t = load_psth_mean(psth_path, 'lick', cond,
                                        baseline_subtract=True)
        except (KeyError, ValueError):
            continue
        traj = project_to_plane(mean, weights, basis)
        plot_trajectory(traj, t, ax, color=color, label=cond_key, lw=1.5)
        ax.plot(traj[0], traj[1], color=color, lw=1.5, ls=ls)
    ax.set_title(f'Lick ({outcome})')
    ax.legend(fontsize=7)


def plot_change_trajectories(psth_path, weights, basis, ax,
                             block='late', outcome='hit'):
    """
    Change-onset trajectories coloured by change TF magnitude.
    Splits into large vs small based on median TF value.
    """
    # find available change conditions for this block/outcome
    with h5py.File(psth_path, 'r') as f:
        ch_conds = sorted(f['ch'].keys())

    prefix = f'{block}_{outcome}_tf'
    matching = [c for c in ch_conds if c.startswith(prefix)]
    if not matching:
        return

    tf_vals = sorted([float(c.split('_tf')[-1]) for c in matching])
    cmap = plt.cm.get_cmap(PLOT_COLOURS['ch_tf_cmap'])
    colors = cmap(np.linspace(0.15, 0.85, len(tf_vals)))

    for tf_val, color in zip(tf_vals, colors):
        cond = f'{prefix}{tf_val}'
        mean, _, t = load_psth_mean(psth_path, 'ch', cond,
                                    baseline_subtract=True)
        traj = project_to_plane(mean, weights, basis)
        plot_trajectory(traj, t, ax, color=color, label=f'tf={tf_val}')
    ax.set_title(f'Change ({block} block, {outcome})')
    ax.legend(fontsize=6)


def plot_session_dynamics(sess_dir, pca_key='event_all',
                          lds_cond='lateBlock_early',
                          plane_method='flow',
                          ops=ANALYSIS_OPTIONS,
                          save_path=None):
    """
    Summary figure for one session: flow fields + trajectory overlays.

    Layout (2 rows x 4 cols):
        row 0: flow field with baseline / TF (x3 conditions)
        row 1: flow field with lick FA / lick hit / change hit / change miss
    """
    import pandas as pd

    sess_data = Session.load(str(sess_dir / 'session.pkl'))
    from utils.filing import load_fr_matrix
    fr_matrix = load_fr_matrix(sess_dir / 'FR_matrix.parquet')
    psth_path = str(sess_dir / 'psths.h5')

    weights, A, basis = _load_weights_and_basis(
        sess_dir, pca_key, lds_cond, fr_matrix, sess_data, ops, plane_method)

    # figure out grid range from actual data
    t_ax = fr_matrix.columns.values
    Z = weights.T @ fr_matrix.values
    proj = basis.T @ Z
    pad = 0.5
    grid_range = (proj.min() - pad, proj.max() + pad)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # row 0: baseline + 3 TF conditions
    plot_flow_field(A, basis, axes[0, 0], grid_range=grid_range)
    plot_baseline_trajectories(psth_path, weights, basis, axes[0, 0])

    for col, bl in enumerate(['earlyBlock_early', 'lateBlock_early',
                               'lateBlock_late'], start=1):
        plot_flow_field(A, basis, axes[0, col], grid_range=grid_range)
        plot_tf_trajectories(psth_path, weights, basis, axes[0, col],
                             block_label=bl)

    # row 1: lick FA, lick hit, change hit, change miss
    plot_flow_field(A, basis, axes[1, 0], grid_range=grid_range)
    plot_lick_trajectories(psth_path, weights, basis, axes[1, 0], outcome='fa')

    plot_flow_field(A, basis, axes[1, 1], grid_range=grid_range)
    plot_lick_trajectories(psth_path, weights, basis, axes[1, 1], outcome='hit')

    plot_flow_field(A, basis, axes[1, 2], grid_range=grid_range)
    plot_change_trajectories(psth_path, weights, basis, axes[1, 2],
                             block='late', outcome='hit')

    plot_flow_field(A, basis, axes[1, 3], grid_range=grid_range)
    plot_change_trajectories(psth_path, weights, basis, axes[1, 3],
                             block='late', outcome='miss')

    for ax in axes.flat:
        ax.set_xlabel('Dynamics dim 1')
        ax.set_ylabel('Dynamics dim 2')

    fig.suptitle(f'{sess_data.animal}_{sess_data.name}  '
                 f'(pca={pca_key}, lds={lds_cond}, plane={plane_method})',
                 fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig
