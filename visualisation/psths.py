"""
Functions for plotting event-aligned PSTHS
"""
import numpy as np
import h5py
# from scipy.ndimage import gaussian_filter1d
from utils.smoothing import causal_boxcar
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set_style("whitegrid")

from config import ANALYSIS_OPTIONS, PATHS, PLOT_COLOURS
from data.session import Session
from analyses.load_responses import load_psth
from utils.filing import get_response_files

from concurrent.futures import ProcessPoolExecutor
from functools import partial

def plot_psth(t_ax: np.ndarray,
              mu: np.ndarray,
              err: np.ndarray,
              ax: plt.Axes,
              color: str = 'k',
              label: str = None,
              ls: str = '-',
              zero_line: bool = True):
    ax.plot(t_ax, mu, color=color, lw=1.5, label=label, ls=ls)
    ax.fill_between(t_ax, mu - err, mu + err, color=color, alpha=0.2)
    if zero_line:
        ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('FR (Hz)')
    if label:
        ax.legend(fontsize=7)


def plot_raster(t_ax: np.ndarray,
                arr: np.ndarray,  # nEv x nT, raw (unsmoothed) for a single unit
                ax: plt.Axes,
                color: str = 'k',
                y_offset: int = 0,
                zero_line: bool = True):
    for i, trial in enumerate(arr):
        ax.vlines(t_ax[trial > 0], y_offset + i + 0.5, y_offset + i + 1.5,
                  color=color, lw=0.5)
    if zero_line:
        ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlim(t_ax[0], t_ax[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')


def plot_grouped_raster(t_ax: np.ndarray,
                        conditions: list[tuple[np.ndarray, str, str]],  # (arr, color, label)
                        ax: plt.Axes,
                        gap: int = 3):
    """
    Plot raster for multiple conditions grouped and sorted, with a small gap between groups.
    arr in each tuple should be raw (unsmoothed), nEv x nT for a single unit.
    """
    y_offset = 0
    for arr, color, label in conditions:
        if arr.shape[0] == 0:
            continue
        plot_raster(t_ax, arr, ax, color=color, y_offset=y_offset, zero_line=False)
        y_offset += arr.shape[0] + gap

    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_ylim(0, y_offset)
    ax.set_xlim(t_ax[0], t_ax[-1])


def _load_condition(psth_path: str,
                    event_type: str,
                    condition: str,
                    unit_idx: int,
                    ops: dict = ANALYSIS_OPTIONS):
    """Load and smooth one condition, return mu, sem, raw arr, t_ax."""
    arr, t = load_psth(psth_path, event_type, condition)
    if arr.shape[0] == 0:
        nT = len(t)
        return np.zeros(nT), np.zeros(nT), np.zeros((0, nT)), t
    sm  = ops['sp_smooth_width'] / ops['sp_bin_width']
    smooth = causal_boxcar(arr, window_bins=sm, axis=-1)
    mu     = np.nanmean(smooth[:, unit_idx, :], axis=0)
    sem    = np.nanstd( smooth[:, unit_idx, :], axis=0) / np.sqrt(arr.shape[0])
    raw    = arr[:, unit_idx, :]  # unsmoothed for raster
    return mu, sem, raw, t


def plot_basic_psths(psth_path: str,
                     unit_idx: int = 0,
                     save_dir: str = None,
                     ops: dict = ANALYSIS_OPTIONS,
                     region: str = None):
    """
    Single-unit summary figure. Layout (2 rows x 7 cols):
      row 0: PSTHs
      row 1: rasters (grouped by condition, sorted)
      col 0: baseline (early/late block)
      col 1: TF pulses, early block, early in trial
      col 2: TF pulses, late block, early in trial
      col 3: TF pulses, late block, late in trial
      col 4: change hits, early block (by TF magnitude)
      col 5: change hits, late block (by TF magnitude)
      col 6: FAs (early block solid, late block dashed)
    """
    ncol = 7
    fig, axes = plt.subplots(2, ncol, figsize=(ncol * 3, 6),
                             gridspec_kw={'hspace': 0.05, 'wspace': 0.4},
                             constrained_layout=True)

    c_block = PLOT_COLOURS['block']
    c_tf    = PLOT_COLOURS['tf']
    u       = unit_idx

    # baseline
    raster_conds = []
    for block, color in c_block.items():
        mu, sem, raw, t = _load_condition(psth_path, 'bl', block, u, ops)
        plot_psth(t, mu, sem, axes[0, 0], color=color, label=block)
        raster_conds.append((raw, color, block))
    plot_grouped_raster(t, raster_conds, axes[1, 0])
    axes[0, 0].set_title('Baseline onset')
    axes[0, 0].legend(fontsize=7)

    # TF pulses
    tf_layout = {
        1: ('earlyBlock', 'early'),
        2: ('lateBlock',  'early'),
        3: ('lateBlock',  'late'),
    }
    tf_titles = {
        1: 'TF early-in-trial\n(early block)',
        2: 'TF early-in-trial\n(late block)',
        3: 'TF late-in-trial\n(late block)',
    }
    for col, (block, tr_phase) in tf_layout.items():
        raster_conds = []
        for polarity, color in c_tf.items():
            cond = f'{block}_{tr_phase}_{polarity}'
            mu, sem, raw, t = _load_condition(psth_path, 'tf', cond, u, ops)
            plot_psth(t, mu, sem, axes[0, col], color=color, label=polarity)
            raster_conds.append((raw, color, polarity))
        plot_grouped_raster(t, raster_conds, axes[1, col])
        axes[0, col].set_title(tf_titles[col])
        axes[0, col].legend(fontsize=7)

    # change hits, one line per TF magnitude
    with h5py.File(psth_path, 'r') as f:
        ch_conds = list(f['ch'].keys())
    ch_tfs = sorted({c.split('_tf')[-1] for c in ch_conds if '_tf' in c}, key=float)
    ch_colors = list(
        plt.cm.get_cmap(PLOT_COLOURS['ch_tf_cmap'])(np.linspace(0.15, 0.85, len(ch_tfs))))
    ch_colors[0] = (0.6, 0.6, 0.6, 1.0)  # grey for no-change (smallest tf)

    for block, col in {'early': 4, 'late': 5}.items():
        raster_conds = []
        for tf_val, color in zip(ch_tfs, ch_colors):
            cond = f'{block}_hit_tf{tf_val}'
            if cond not in ch_conds:
                continue
            mu, sem, raw, t = _load_condition(psth_path, 'ch', cond, u, ops)
            plot_psth(t, mu, sem, axes[0, col], color=color, label=f'tf={tf_val}')
            raster_conds.append((raw, color, f'tf={tf_val}'))
        plot_grouped_raster(t, raster_conds, axes[1, col])
        axes[0, col].set_title(f'Change hits\n({block} block)')
        axes[0, col].legend(fontsize=7)

    # FAs: early/late block by color, early/late in trial solid/dashed
    fa_layout = [
        ('earlyBlock', 'early', '-'),
        ('lateBlock',  'early', '-'),
        ('lateBlock',  'late',  '--'),
    ]
    raster_conds = []
    for block, tr_phase, ls in fa_layout:
        color = c_block['early' if block == 'earlyBlock' else 'late']
        cond  = f'{block}_{tr_phase}_fa'
        mu, sem, raw, t = _load_condition(psth_path, 'lick', cond, u, ops)
        plot_psth(t, mu, sem, axes[0, 6], color=color, label=f'{block} {tr_phase}', ls=ls)
        raster_conds.append((raw, color, f'{block} {tr_phase}'))
    plot_grouped_raster(t, raster_conds, axes[1, 6])
    axes[0, 6].set_title('FAs')
    axes[0, 6].legend(fontsize=7)

    # tidy up
    for col in range(ncol):
        for row in range(2):
            axes[row, col].axvline(0, color='gray', lw=0.8, ls='--')
            axes[row, col].spines[['top', 'right']].set_visible(False)
        if col > 0:
            axes[0, col].set_ylabel('')
            axes[1, col].set_ylabel('')

    title = f'Unit {unit_idx}' + (f' ({region})' if region else '')
    fig.suptitle(title, fontsize=11)

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / f'unit_{unit_idx:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def _plot_unit(unit_idx: int,
               psth_path: str,
               save_dir: str,
               ops: dict,
               region: str = None):
    """Worker fn for parallelising across units."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
            plot_basic_psths(psth_path, unit_idx=unit_idx,
                             save_dir=save_dir, ops=ops, region=region)
    except Exception as e:
        print(f'unit {unit_idx} failed: {e}')
    finally:
        plt.close('all')


def plot_all_su_psths(npx_dir: str = PATHS['npx_dir_local'],
                      plots_dir: str = PATHS['plots_dir'],
                      ops: dict = ANALYSIS_OPTIONS,
                      n_workers: int = 8):
    """
    Runs through all units in all sessions to generate basic single unit PSTHs.
    Saves figures to plots_dir,
    """
    psth_paths = get_response_files(npx_dir)

    for i, psth_path in enumerate(psth_paths):
        sess_data = Session.load(psth_path.replace('psths.h5', 'session.pkl'))
        print(f'{sess_data.animal}_{sess_data.name} '
              f'({i + 1}/{len(psth_paths)}, {sess_data.n_neurons} units)')
        save_dir  = str(Path(plots_dir) / sess_data.animal / sess_data.name / 'su_psths')
        regions = sess_data.unit_info['brain_region_comb'].values

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_plot_unit, unit_idx=i, psth_path=psth_path,
                                   save_dir=save_dir, ops=ops, region=regions[i])
                       for i in range(sess_data.n_neurons)]
            for f in futures:
                f.result()