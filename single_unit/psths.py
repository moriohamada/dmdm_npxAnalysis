"""
Per-unit event-aligned PSTH/raster summary figures from session psths.h5 files.
"""

from config import ANALYSIS_OPTIONS, PATHS, PLOT_OPTIONS
from data.session import Session
from utils.smoothing import causal_boxcar
from utils.filing import get_response_files

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import warnings
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


def plot_psth(t_ax: np.ndarray,
              mu: np.ndarray,
              err: np.ndarray,
              ax: plt.Axes,
              color: str = 'k',
              label: str = None,
              ls: str = '-',
              zero_line: bool = True):
    """plot mean +/- err on ax"""
    ax.plot(t_ax, mu, color=color, lw=1.5, label=label, ls=ls)
    ax.fill_between(t_ax, mu - err, mu + err, color=color, alpha=0.2)
    if zero_line:
        ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('FR (Hz)')
    if label:
        ax.legend(fontsize=7)


def plot_raster(t_ax: np.ndarray,
                arr: np.ndarray,
                ax: plt.Axes,
                color: str = 'k',
                y_offset: int = 0,
                zero_line: bool = True):
    """raster for one unit; arr is nEv x nT raw spikes"""
    for i, trial in enumerate(arr):
        ax.vlines(t_ax[trial > 0], y_offset + i + 0.5, y_offset + i + 1.5,
                  color=color, lw=0.5)
    if zero_line:
        ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlim(t_ax[0], t_ax[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')


def plot_grouped_raster(t_ax: np.ndarray,
                        conditions: list[tuple[np.ndarray, str, str]],
                        ax: plt.Axes,
                        gap: int = 3):
    """
    stack rasters of multiple conditions vertically with a gap between groups.
    conditions: list of (arr, color, label), arr is nEv x nT raw spikes for one unit.
    """
    y_offset = 0
    for arr, color, _ in conditions:
        if arr.shape[0] == 0:
            continue
        plot_raster(t_ax, arr, ax, color=color, y_offset=y_offset, zero_line=False)
        y_offset += arr.shape[0] + gap

    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_ylim(0, y_offset)
    ax.set_xlim(t_ax[0], t_ax[-1])


def _extract_unit(f: h5py.File,
                  event_type: str,
                  condition: str,
                  unit_idx: int,
                  ops: dict = ANALYSIS_OPTIONS):
    """
    pull one unit's data for one (event_type, condition) from an open psths.h5
    Returns:
        mu, sem: nT smoothed mean and SEM across events
        raw:     nEv x nT unsmoothed spikes
        t:       time axis (post smoothing-buffer trim)
    """
    key = f'{event_type}/{condition}'
    t = f[f't_ax/{event_type}'][:]
    if key not in f or f[key].shape[0] == 0:
        nT = len(t)
        return np.zeros(nT), np.zeros(nT), np.zeros((0, nT)), t

    raw = f[key][:, unit_idx, :]  # nEv x nT
    sm_bins = 2 * ops['sp_smooth_width'] / ops['sp_bin_width']
    smooth = causal_boxcar(raw, window_bins=sm_bins, axis=-1)
    mu = np.nanmean(smooth, axis=0)
    sem = np.nanstd(smooth, axis=0) / np.sqrt(raw.shape[0])

    # trim smoothing edge
    buf = ops.get('resp_buffer', 0)
    if buf > 0:
        keep = t >= t[0] + buf
        t, mu, sem, raw = t[keep], mu[keep], sem[keep], raw[:, keep]

    return mu, sem, raw, t


def plot_basic_psths(psth_path: str,
                     unit_idx: int = 0,
                     save_dir: str = None,
                     ops: dict = ANALYSIS_OPTIONS,
                     region: str = None):
    """
    summary figure for one unit, 2 rows x 7 cols (PSTH on top, raster below):
        col 0: baseline, by block
        col 1: TF pulses, early-in-trial, early block
        col 2: TF pulses, early-in-trial, late block
        col 3: TF pulses, late-in-trial,  late block
        col 4: change hits, early block (one line per TF magnitude)
        col 5: change hits, late block
        col 6: FAs (block as colour, in-trial position as solid/dashed)
    """
    ncol = 7
    fig, axes = plt.subplots(2, ncol, figsize=(ncol * 3, 6),
                             gridspec_kw={'hspace': 0.05, 'wspace': 0.4},
                             constrained_layout=True)

    c_block = PLOT_OPTIONS['colours']['block']
    c_tf    = PLOT_OPTIONS['colours']['tf']

    with h5py.File(psth_path, 'r') as f:

        # baseline
        rasters = []
        for block, color in c_block.items():
            mu, sem, raw, t = _extract_unit(f, 'bl', block, unit_idx, ops)
            plot_psth(t, mu, sem, axes[0, 0], color=color, label=block)
            rasters.append((raw, color, block))
        plot_grouped_raster(t, rasters, axes[1, 0])
        axes[0, 0].set_title('Baseline onset')
        axes[0, 0].legend(fontsize=7)

        # TF pulses
        tf_layout = {
            1: ('earlyBlock', 'early', 'TF early-in-trial\n(early block)'),
            2: ('lateBlock',  'early', 'TF early-in-trial\n(late block)'),
            3: ('lateBlock',  'late',  'TF late-in-trial\n(late block)'),
        }
        for col, (block, tr_phase, title) in tf_layout.items():
            rasters = []
            for polarity, color in c_tf.items():
                cond = f'{block}_{tr_phase}_{polarity}'
                mu, sem, raw, t = _extract_unit(f, 'tf', cond, unit_idx, ops)
                plot_psth(t, mu, sem, axes[0, col], color=color, label=polarity)
                rasters.append((raw, color, polarity))
            plot_grouped_raster(t, rasters, axes[1, col])
            axes[0, col].set_title(title)
            axes[0, col].legend(fontsize=7)

        # change hits, one line per TF magnitude
        ch_conds = list(f['ch'].keys())
        ch_tfs = sorted({c.split('_tf')[-1] for c in ch_conds if '_tf' in c}, key=float)
        ch_colors = list(plt.cm.get_cmap(PLOT_OPTIONS['colours']['ch_tf_cmap'])(
            np.linspace(0.15, 0.85, len(ch_tfs))))
        ch_colors[0] = (0.6, 0.6, 0.6, 1.0)  # smallest TF in grey

        for block, col in {'early': 4, 'late': 5}.items():
            rasters = []
            for tf_val, color in zip(ch_tfs, ch_colors):
                cond = f'{block}_hit_tf{tf_val}'
                if cond not in ch_conds:
                    continue
                mu, sem, raw, t = _extract_unit(f, 'ch', cond, unit_idx, ops)
                plot_psth(t, mu, sem, axes[0, col], color=color, label=f'tf={tf_val}')
                rasters.append((raw, color, f'tf={tf_val}'))
            plot_grouped_raster(t, rasters, axes[1, col])
            axes[0, col].set_title(f'Change hits\n({block} block)')
            axes[0, col].legend(fontsize=7)

        # FAs
        fa_layout = [
            ('earlyBlock', 'early', '-'),
            ('lateBlock',  'early', '-'),
            ('lateBlock',  'late',  '--'),
        ]
        rasters = []
        for block, tr_phase, ls in fa_layout:
            color = c_block['early' if block == 'earlyBlock' else 'late']
            cond  = f'{block}_{tr_phase}_fa'
            mu, sem, raw, t = _extract_unit(f, 'lick', cond, unit_idx, ops)
            plot_psth(t, mu, sem, axes[0, 6], color=color, ls=ls,
                      label=f'{block} {tr_phase}')
            rasters.append((raw, color, f'{block} {tr_phase}'))
        plot_grouped_raster(t, rasters, axes[1, 6])
        axes[0, 6].set_title('FAs')
        axes[0, 6].legend(fontsize=7)

    # tidy
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
    """worker for parallel per-unit plotting"""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
            plot_basic_psths(psth_path, unit_idx=unit_idx,
                             save_dir=save_dir, ops=ops, region=region)
    except Exception as e:
        print(f'      unit {unit_idx} failed: {e}')
    finally:
        plt.close('all')


def plot_all_su_psths(npx_dir: str = PATHS['npx_dir_local'],
                      plots_dir: str = PATHS['plots_dir'],
                      ops: dict = ANALYSIS_OPTIONS,
                      n_workers: int = 8):
    """
    Runner to loop through all sessions and plot per-unit summary PSTHs in parallel.
    Saves one png per unit under <plots_dir>/<animal>/<session>/su_psths/.
    """
    psth_paths = get_response_files(npx_dir)

    for i, psth_path in enumerate(psth_paths):
        sess = Session.load(psth_path.replace('psths.h5', 'session.pkl'))
        print(f'{sess.animal}_{sess.name} '
              f'({i + 1}/{len(psth_paths)}, {sess.n_neurons} units)')
        save_dir = str(Path(plots_dir) / sess.animal / sess.name / 'su_psths')
        regions = sess.unit_info['brain_region_comb'].values

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            jobs = [pool.submit(_plot_unit, unit_idx=u, psth_path=psth_path,
                                save_dir=save_dir, ops=ops, region=regions[u])
                    for u in range(sess.n_neurons)]
            for job in jobs:
                job.result()


if __name__ == '__main__':
    plot_all_su_psths()