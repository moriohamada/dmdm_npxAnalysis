"""
plotting for block modulation analyses
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

sns.set_style("whitegrid")

from config import PATHS, ANALYSIS_OPTIONS, PLOT_OPTIONS
from data.load_responses import load_psth_mean
from utils.filing import get_response_files
from utils.smoothing import causal_boxcar


EARLY_COL = PLOT_OPTIONS['colours']['block']['early']
LATE_COL = PLOT_OPTIONS['colours']['block']['late']
FAST_COL = PLOT_OPTIONS['colours']['tf_pref']['fast']
SLOW_COL = PLOT_OPTIONS['colours']['tf_pref']['slow']


#%% loading helpers

def _load_all_tuning_results(npx_dir):
    """load per-session tuning_curves.pkl files"""
    psth_paths = get_response_files(npx_dir)
    all_results = []
    for path in psth_paths:
        pkl_path = Path(path).parent / 'tuning_curves.pkl'
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                all_results.append(pickle.load(f))
    return all_results


#%% analysis 1: tuning curves

PSTH_SMOOTH_BINS = int(round(
    2 * ANALYSIS_OPTIONS['sp_smooth_width'] / ANALYSIS_OPTIONS['sp_bin_width']))


def _get_group_averages(data, group_indices, unit_mask=None):
    """
    compute population mean and per-group means, optionally filtered by unit_mask.
    data: (n_bins, n_units)
    returns pop_mean (n_bins,), group_means list of (n_bins,)
    """
    if unit_mask is not None:
        valid = np.where(unit_mask)[0]
        valid_set = set(valid.tolist())
        remap = {old: new for new, old in enumerate(valid)}
        d = data[:, unit_mask]
        groups = []
        for idx in group_indices:
            remapped = [remap[i] for i in idx if i in valid_set]
            if remapped:
                groups.append(np.nanmean(d[:, remapped], axis=1))
    else:
        d = data
        groups = [np.nanmean(d[:, idx], axis=1) for idx in group_indices]
    pop = np.nanmean(d, axis=1)
    return pop, groups


def _draw_panel(ax, x, pop_mean, group_means, colour, label=None):
    """draw thin group lines + bold population mean on a single axis"""
    thin_alpha = min(0.3, max(0.02, 5.0 / max(len(group_means), 1)))
    for gm in group_means:
        ax.plot(x, gm, color=colour, alpha=thin_alpha, linewidth=0.7)
    ax.plot(x, pop_mean, color=colour, linewidth=2.5, label=label)

    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')


def _make_tuning_figure(binned, bin_centres, group_indices, sig_mask,
                         mean_gains, title, save_path=None):
    """
    3 rows x 4 cols tuning curve summary.
    cols 0-1: all units (raw, normalised)
    cols 2-3: TF-responsive only, split by fast/slow pref (raw, normalised)
    rows: early, late, difference
    """
    x = bin_centres
    n_units = binned['early'].shape[1]
    n_sig = sig_mask.sum()
    fast_mask = sig_mask & (mean_gains > 0)
    slow_mask = sig_mask & (mean_gains < 0)
    n_fast, n_slow = fast_mask.sum(), slow_mask.sum()

    # normalised
    norm = {}
    for block in ['early', 'late']:
        norm[block] = binned[block] - np.nanmean(binned[block], axis=0, keepdims=True)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))

    row_labels = ['Early block', 'Late block', 'Difference (early - late)']
    col_labels = ['All units', 'All units (normalised)',
                  f'TF-responsive (n={n_sig})',
                  f'TF-responsive (normalised)']
    block_colours = {'early': EARLY_COL, 'late': LATE_COL, 'diff': 'black'}

    for col in range(4):
        is_norm = col % 2 == 1
        is_sig = col >= 2
        data_dict = norm if is_norm else binned

        for row in range(3):
            ax = axes[row, col]

            if row < 2:
                block = 'early' if row == 0 else 'late'
                data = data_dict[block]
            else:
                data = data_dict['early'] - data_dict['late']

            if not is_sig:
                colour = block_colours.get(
                    'early' if row == 0 else ('late' if row == 1 else 'diff'))
                pop, groups = _get_group_averages(data, group_indices)
                _draw_panel(ax, x, pop, groups, colour)
            else:
                if n_fast > 0:
                    pop, groups = _get_group_averages(data, group_indices, fast_mask)
                    _draw_panel(ax, x, pop, groups, FAST_COL,
                                label=f'Fast (n={n_fast})' if row == 0 else None)
                if n_slow > 0:
                    pop, groups = _get_group_averages(data, group_indices, slow_mask)
                    _draw_panel(ax, x, pop, groups, SLOW_COL,
                                label=f'Slow (n={n_slow})' if row == 0 else None)
                if row == 0:
                    ax.legend(fontsize=7)

            if row == 0:
                ax.set_title(col_labels[col])
            if col == 0:
                ax.set_ylabel(row_labels[row])
            if row == 2:
                ax.set_xlabel('TF (log2 octaves)')

    fig.suptitle(f'{title} (n={n_units}, {n_fast} fast-pref, {n_slow} slow-pref)')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_tuning_curves(npx_dir=PATHS['npx_dir_local'], save_dir=None):
    """
    tuning curve summary figures at per-session, per-animal, and grand average levels
    """
    all_results = _load_all_tuning_results(npx_dir)
    if not all_results:
        return
    save_dir = Path(save_dir) if save_dir else None

    # collect pre-binned data from each session
    session_binned = []
    session_bin_centres = []
    session_sig = []
    session_gains = []
    session_meta = []

    for result in all_results:
        session_binned.append(result['binned'])
        session_bin_centres.append(result['bin_centres'])
        session_sig.append(result['gain_p_tf'] < 0.05)
        session_gains.append(result['gain_pooled'])
        session_meta.append((result['animal'], result['session']))

    # use mean bin centres across sessions as shared x-axis
    bin_centres = np.nanmean(session_bin_centres, axis=0)

    # --- per-session ---
    for i, (binned, sig, gains, (animal, sess)) in enumerate(
            zip(session_binned, session_sig, session_gains, session_meta)):
        n_units = binned['early'].shape[1]
        group_idx = [np.array([u]) for u in range(n_units)]
        sp = save_dir / animal / sess / 'tuning_curves.png' if save_dir else None
        fig = _make_tuning_figure(binned, session_bin_centres[i], group_idx,
                                   sig, gains, f'{animal}/{sess}', sp)
        plt.close(fig)

    # --- per-animal ---
    animals = {}
    for i, (animal, _) in enumerate(session_meta):
        animals.setdefault(animal, []).append(i)

    animal_binned = {}
    animal_sig = {}
    animal_gains = {}
    animal_groups = {}

    for animal, sess_idx_list in animals.items():
        ab = {block: np.concatenate([session_binned[i][block] for i in sess_idx_list], axis=1)
              for block in ['early', 'late']}
        a_sig = np.concatenate([session_sig[i] for i in sess_idx_list])
        a_gains = np.concatenate([session_gains[i] for i in sess_idx_list])

        groups = []
        offset = 0
        for i in sess_idx_list:
            n = session_binned[i]['early'].shape[1]
            groups.append(np.arange(offset, offset + n))
            offset += n

        animal_binned[animal] = ab
        animal_sig[animal] = a_sig
        animal_gains[animal] = a_gains
        animal_groups[animal] = groups

        sp = save_dir / animal / 'tuning_curves.png' if save_dir else None
        fig = _make_tuning_figure(ab, bin_centres, groups, a_sig, a_gains,
                                   animal, sp)
        plt.close(fig)

    # --- grand average ---
    animal_list = list(animals.keys())
    grand = {block: np.concatenate([animal_binned[a][block] for a in animal_list], axis=1)
             for block in ['early', 'late']}
    grand_sig = np.concatenate([animal_sig[a] for a in animal_list])
    grand_gains = np.concatenate([animal_gains[a] for a in animal_list])

    grand_groups = []
    offset = 0
    for a in animal_list:
        n = animal_binned[a]['early'].shape[1]
        grand_groups.append(np.arange(offset, offset + n))
        offset += n

    sp = save_dir / 'tuning_curves_grand.png' if save_dir else None
    fig = _make_tuning_figure(grand, bin_centres, grand_groups, grand_sig,
                               grand_gains, 'Grand average', sp)
    plt.close(fig)


#%% per-neuron tuning figures

def _plot_single_neuron(n, tc, tc_sem, bin_centres, tf_psths, fa_psths,
                         unit_info, gains, offsets, gain_p_block, gain_diff_p,
                         offset_diff_p, save_path=None):
    """
    single neuron figure: tuning curve + TF PSTH + FA PSTH.
    tc: {block: (n_bins, nN)} pre-binned tuning curves
    tc_sem: {block: (n_bins, nN)} SEM per bin
    bin_centres: (n_bins,) median TF per bin
    gains: (nN, 2) per-block gains, offsets: (nN, 2) per-block offsets
    gain_p_block: (nN, 2) per-block gain p-values
    gain_diff_p: (nN,) gain block difference p-values
    offset_diff_p: (nN,) offset block difference p-values
    """
    area = unit_info.iloc[n].get('brain_region_comb', '?') if hasattr(unit_info, 'iloc') else '?'
    pref = 'fast' if (gains[n, 0] + gains[n, 1]) > 0 else 'slow'

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # panel 0: tuning curve with SEM bands and linear fits
    ax = axes[0]
    for block, colour, bi in [('early', EARLY_COL, 0), ('late', LATE_COL, 1)]:
        y = tc[block][:, n]
        sem = tc_sem[block][:, n]
        ax.plot(bin_centres, y, color=colour, linewidth=1.5, label=f'{block} block')
        ax.fill_between(bin_centres, y - sem, y + sem, color=colour, alpha=0.2)
        fit_y = gains[n, bi] * bin_centres + offsets[n, bi]
        ax.plot(bin_centres, fit_y, color=colour, linewidth=1, linestyle='--')

    ax.set_xlabel('TF (log2 octaves)')
    ax.set_ylabel('Response (z-scored FR)')
    ax.set_title('TF tuning curve')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=7)

    # panel 1: TF-pulse-aligned PSTH (fast minus slow)
    ax = axes[1]
    if tf_psths is not None:
        t = tf_psths['t_ax']
        for block, colour in [('early', EARLY_COL), ('late', LATE_COL)]:
            pos = tf_psths[f'{block}_pos'][n]
            neg = tf_psths[f'{block}_neg'][n]
            pos_sem = tf_psths[f'{block}_pos_sem'][n]
            neg_sem = tf_psths[f'{block}_neg_sem'][n]
            mu = pos - neg
            sem = np.sqrt(pos_sem**2 + neg_sem**2)
            ax.plot(t, mu, color=colour, lw=1.5)
            ax.fill_between(t, mu - sem, mu + sem, color=colour, alpha=0.2)
    ax.axvline(0, color='grey', lw=0.8, ls='--')
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlim(-0.5, 1.0)
    ax.set_xlabel('Time from TF pulse (s)')
    ax.set_title('TF PSTH (fast - slow)')

    # panel 2: FA-aligned PSTH
    ax = axes[2]
    if fa_psths is not None:
        t = fa_psths['t_ax']
        for block, colour in [('early', EARLY_COL), ('late', LATE_COL)]:
            if fa_psths[block] is not None:
                mu = fa_psths[block][n]
                sem = fa_psths[block + '_sem'][n]
                ax.plot(t, mu, color=colour, lw=1.5)
                ax.fill_between(t, mu - sem, mu + sem, color=colour, alpha=0.2)
    ax.axvline(0, color='grey', lw=0.8, ls='--')
    ax.set_xlim(-1.5, 0.5)
    ax.set_xlabel('Time from FA lick (s)')
    ax.set_title('FA-aligned PSTH')

    fig.suptitle(
        f'{area} | unit {n} | early: g={gains[n,0]:.3f} (p={gain_p_block[n,0]:.3f}) | '
        f'late: g={gains[n,1]:.3f} (p={gain_p_block[n,1]:.3f}) | '
        f'gain diff p={gain_diff_p[n]:.3f} | offset diff p={offset_diff_p[n]:.3f} | '
        f'{pref}-pref',
        fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _load_session_psths(sess_dir):
    """
    load TF and FA PSTHs (mean + SEM) from psths.h5 for per-neuron plots.
    TF PSTHs loaded separately for pos and neg to avoid cancellation.
    """
    psth_path = str(Path(sess_dir) / 'psths.h5')
    buf = ANALYSIS_OPTIONS.get('resp_buffer', 0)

    def _smooth_and_trim(mean, sem, t_ax):
        """smooth causally and trim buffer from start"""
        mean = causal_boxcar(mean, PSTH_SMOOTH_BINS, axis=-1)
        sem = causal_boxcar(sem, PSTH_SMOOTH_BINS, axis=-1)
        if buf > 0:
            trim = t_ax >= t_ax[0] + buf
            mean, sem, t_ax = mean[:, trim], sem[:, trim], t_ax[trim]
        return mean, sem, t_ax

    # TF PSTHs: load pos and neg separately per block
    tf_psths = None
    try:
        t_tf = None
        tf_data = {}
        for block_prefix in ['earlyBlock', 'lateBlock']:
            block = 'early' if block_prefix == 'earlyBlock' else 'late'
            for pol in ['pos', 'neg']:
                cond = f'{block_prefix}_early_{pol}'
                m, s, t = load_psth_mean(psth_path, 'tf', cond)
                if m is None:
                    raise ValueError(f'no events for {cond}')
                m, s, t = _smooth_and_trim(m, s, t)
                tf_data[f'{block}_{pol}'] = m
                tf_data[f'{block}_{pol}_sem'] = s
                if t_tf is None:
                    t_tf = t
        tf_psths = {**tf_data, 't_ax': t_tf}
    except (KeyError, ValueError):
        pass

    # FA PSTHs by block
    fa_psths = None
    fa_data = {}
    t_fa = None
    for block, cond in [('early', 'earlyBlock_early_fa'),
                         ('late', 'lateBlock_early_fa')]:
        try:
            m, s, t = load_psth_mean(psth_path, 'lick', cond)
            if m is None:
                continue
            m, s, t = _smooth_and_trim(m, s, t)
            fa_data[block] = m
            fa_data[block + '_sem'] = s
            if t_fa is None:
                t_fa = t
        except (KeyError, ValueError):
            fa_data[block] = None
            fa_data[block + '_sem'] = None

    if t_fa is not None:
        fa_psths = {**fa_data, 't_ax': t_fa}

    return tf_psths, fa_psths


def plot_session_su_tuning(result, sess_dir, save_dir=None):
    """
    plot per-neuron figures for a single session.
    called on the fly during extraction with the result dict and session directory.
    """
    save_dir = Path(save_dir) if save_dir else None
    animal = result['animal']
    sess = result['session']
    tc = result['binned']
    tc_sem = result['binned_sem']
    bin_centres = result['bin_centres']
    unit_info = result['unit_info']
    gains = result['gain']
    offsets = result['offset']
    gain_p_block = result['gain_p_block']
    gain_diff_p = result['gain_diff_p']
    offset_diff_p = result['offset_diff_p']
    n_neurons = gains.shape[0]

    tf_psths, fa_psths = _load_session_psths(sess_dir)

    for n in range(n_neurons):
        sp = (save_dir / 'su_tuning' / animal / sess / f'unit_{n:04d}.png'
              if save_dir else None)
        _plot_single_neuron(n, tc, tc_sem, bin_centres, tf_psths, fa_psths,
                             unit_info, gains, offsets, gain_p_block,
                             gain_diff_p, offset_diff_p, sp)


def plot_single_unit_tuning(npx_dir=PATHS['npx_dir_local'], save_dir=None):
    """
    plot per-neuron figures for all sessions.
    for batch use after extraction (on-the-fly uses plot_session_su_tuning instead).
    """
    all_results = _load_all_tuning_results(npx_dir)
    if not all_results:
        return
    save_dir = Path(save_dir) if save_dir else None
    psth_paths = get_response_files(npx_dir)

    for result in all_results:
        animal = result['animal']
        sess = result['session']
        sess_dir = None
        for p in psth_paths:
            if animal in p and sess in p:
                sess_dir = Path(p).parent
                break
        if sess_dir is None:
            continue
        plot_session_su_tuning(result, sess_dir, save_dir)


def plot_gain_offset_distributions(npx_dir=PATHS['npx_dir_local'], save_dir=None):
    """
    gain and offset summaries.
    row 1: all units - gain diff histogram, gain scatter, offset diff histogram
    row 2: TF-responsive only - gain scatter, offset scatter, gain/offset diff by pref
    """
    all_results = _load_all_tuning_results(npx_dir)
    all_gain = []
    all_offset = []
    all_gain_diff_p = []
    all_offset_diff_p = []
    all_gain_p_tf = []
    all_gain_pooled = []

    for result in all_results:
        all_gain.append(result['gain'])
        all_offset.append(result['offset'])
        all_gain_diff_p.append(result['gain_diff_p'])
        all_offset_diff_p.append(result['offset_diff_p'])
        all_gain_p_tf.append(result['gain_p_tf'])
        all_gain_pooled.append(result['gain_pooled'])

    gain = np.concatenate(all_gain)            # (nN_total, 2)
    offset = np.concatenate(all_offset)
    gain_diff_p = np.concatenate(all_gain_diff_p)
    offset_p = np.concatenate(all_offset_diff_p)
    gain_p_tf = np.concatenate(all_gain_p_tf)
    gain_pooled = np.concatenate(all_gain_pooled)
    gain_diff = gain[:, 0] - gain[:, 1]
    offset_diff = offset[:, 0] - offset[:, 1]

    # all-units masks (block diff significance)
    sig = gain_diff_p < 0.05
    nonsig = ~sig
    mean_gains = np.mean(gain, axis=1)
    fast_sig = sig & (mean_gains > 0)
    slow_sig = sig & (mean_gains < 0)

    # TF-responsive masks (pooled gain significance + pref from pooled gain)
    tf_resp = gain_p_tf < 0.05
    fast_resp = tf_resp & (gain_pooled > 0)
    slow_resp = tf_resp & (gain_pooled < 0)
    n_tf = tf_resp.sum()
    n_fast = fast_resp.sum()
    n_slow = slow_resp.sum()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    #%% row 1: all units

    # top left: gain difference histogram
    ax = axes[0, 0]
    ax.hist(gain_diff, bins=50, color='grey', alpha=0.7, edgecolor='black',
            linewidth=0.5)
    if sig.any():
        ax.hist(gain_diff[sig], bins=50, color=EARLY_COL, alpha=0.7,
                label=f'p < 0.05 (n={sig.sum()})')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(np.nanmedian(gain_diff), color='red', linewidth=1.5,
               label=f'median = {np.nanmedian(gain_diff):.4f}')
    ax.set_xlabel('Gain difference (early - late)')
    ax.set_ylabel('Number of units')
    ax.set_title('All units: gain block difference')
    ax.legend(fontsize=8)

    # top centre: early vs late gain scatter (all units)
    ax = axes[0, 1]
    nonsig_diffs = gain_diff[nonsig]
    if len(nonsig_diffs) > 10:
        band = np.nanpercentile(np.abs(nonsig_diffs), 95)
    else:
        band = np.nanstd(gain_diff) * 1.96

    ax.scatter(gain[nonsig, 0], gain[nonsig, 1], s=5, alpha=0.15,
               color='grey', label=f'n.s. (n={nonsig.sum()})', zorder=1)
    if fast_sig.any():
        ax.scatter(gain[fast_sig, 0], gain[fast_sig, 1], s=12, alpha=0.5,
                   color=FAST_COL, label=f'Fast sig (n={fast_sig.sum()})', zorder=3)
    if slow_sig.any():
        ax.scatter(gain[slow_sig, 0], gain[slow_sig, 1], s=12, alpha=0.5,
                   color=SLOW_COL, label=f'Slow sig (n={slow_sig.sum()})', zorder=3)

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.fill_between(lims, [l - band for l in lims], [l + band for l in lims],
                    color='grey', alpha=0.1, zorder=0, label='95% null band')
    ax.plot(lims, lims, color='black', linewidth=1, linestyle='--', zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Gain (early block)')
    ax.set_ylabel('Gain (late block)')
    ax.set_title(f'All units: gain early vs late (n={len(gain)})')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper left')

    # top right: offset difference histogram
    ax = axes[0, 2]
    ax.hist(offset_diff, bins=50, color='grey', alpha=0.7, edgecolor='black',
            linewidth=0.5)
    sig_off = offset_p < 0.05
    if sig_off.any():
        ax.hist(offset_diff[sig_off], bins=50, color=EARLY_COL, alpha=0.7,
                label=f'p < 0.05 (n={sig_off.sum()})')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(np.nanmedian(offset_diff), color='red', linewidth=1.5,
               label=f'median = {np.nanmedian(offset_diff):.4f}')
    ax.set_xlabel('Offset difference (early - late)')
    ax.set_ylabel('Number of units')
    ax.set_title('All units: offset block difference')
    ax.legend(fontsize=8)

    #%% row 2: TF-responsive units only

    # helper for early-vs-late scatter with fast/slow colouring
    def _scatter_early_late(ax, vals, fast_mask, slow_mask, xlabel, ylabel, title):
        if fast_mask.any():
            ax.scatter(vals[fast_mask, 0], vals[fast_mask, 1], s=15, alpha=0.5,
                       color=FAST_COL, label=f'Fast (n={fast_mask.sum()})', zorder=2)
        if slow_mask.any():
            ax.scatter(vals[slow_mask, 0], vals[slow_mask, 1], s=15, alpha=0.5,
                       color=SLOW_COL, label=f'Slow (n={slow_mask.sum()})', zorder=2)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, color='black', linewidth=1, linestyle='--', zorder=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.legend(fontsize=7)

    # bottom left: gain scatter (TF-responsive)
    ax = axes[1, 0]
    _scatter_early_late(ax, gain[tf_resp], fast_resp[tf_resp], slow_resp[tf_resp],
                        'Gain (early block)', 'Gain (late block)',
                        f'TF-responsive: gain (n={n_tf})')

    # bottom centre: offset scatter (TF-responsive)
    ax = axes[1, 1]
    _scatter_early_late(ax, offset[tf_resp], fast_resp[tf_resp], slow_resp[tf_resp],
                        'Offset (early block)', 'Offset (late block)',
                        f'TF-responsive: offset (n={n_tf})')

    # bottom right: gain and offset diff distributions by pref
    ax = axes[1, 2]
    if n_fast > 0:
        ax.hist(gain_diff[fast_resp], bins=30, color=FAST_COL, alpha=0.4,
                edgecolor=FAST_COL, linewidth=0.5,
                label=f'Gain diff - fast (n={n_fast})')
    if n_slow > 0:
        ax.hist(gain_diff[slow_resp], bins=30, color=SLOW_COL, alpha=0.4,
                edgecolor=SLOW_COL, linewidth=0.5,
                label=f'Gain diff - slow (n={n_slow})')
    if n_fast > 0:
        ax.hist(offset_diff[fast_resp], bins=30, color=FAST_COL, alpha=0.4,
                edgecolor=FAST_COL, linewidth=0.5, linestyle='--', hatch='//',
                label=f'Offset diff - fast')
    if n_slow > 0:
        ax.hist(offset_diff[slow_resp], bins=30, color=SLOW_COL, alpha=0.4,
                edgecolor=SLOW_COL, linewidth=0.5, linestyle='--', hatch='//',
                label=f'Offset diff - slow')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Difference (early - late)')
    ax.set_ylabel('Number of units')
    ax.set_title('TF-responsive: gain & offset diffs by pref')
    ax.legend(fontsize=7)

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / 'gain_offset_distributions.png',
                    dpi=300, bbox_inches='tight')

    return fig


#%% analysis 2: coding dimension rotation

def plot_coding_rotation(npx_dir=PATHS['npx_dir_local'], save_dir=None):
    """
    plot time-resolved cosine similarity and angle between block coding vectors.
    per-mouse lines + average.
    """
    with open(Path(npx_dir) / 'block_modulation' / 'coding_rotation.pkl', 'rb') as f:
        results = pickle.load(f)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    animals = list(results.keys())

    # top left: cosine similarity vs time
    ax = axes[0, 0]
    all_cosine = []
    for animal in animals:
        r = results[animal]
        ax.plot(r['t_ax'], r['cosine_sim'], alpha=0.3, linewidth=0.8)
        all_cosine.append(r['cosine_sim'])
    mean_cosine = np.nanmean(all_cosine, axis=0)
    ax.plot(results[animals[0]]['t_ax'], mean_cosine, color='black',
            linewidth=2, label='Mean')

    # null distribution bands
    all_null = np.concatenate([r['null_cosine'] for r in results.values()], axis=0)
    null_mean = np.nanmean(all_null, axis=0)
    null_ci = np.nanpercentile(all_null, [2.5, 97.5], axis=0)
    t_ax = results[animals[0]]['t_ax']
    ax.fill_between(t_ax, null_ci[0], null_ci[1], color='grey', alpha=0.2,
                    label='95% null CI')
    ax.plot(t_ax, null_mean, color='grey', linewidth=1, linestyle='--')

    ax.set_xlabel('Post-pulse time (s)')
    ax.set_ylabel('Cosine similarity')
    ax.set_title('TF coding direction: between-block similarity')
    ax.legend(fontsize=8)
    ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')

    # top right: angle vs time
    ax = axes[0, 1]
    all_angle = []
    for animal in animals:
        r = results[animal]
        ax.plot(r['t_ax'], r['angle'], alpha=0.3, linewidth=0.8)
        all_angle.append(r['angle'])
    mean_angle = np.nanmean(all_angle, axis=0)
    ax.plot(t_ax, mean_angle, color='black', linewidth=2, label='Mean')
    ax.set_xlabel('Post-pulse time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('TF coding direction: between-block angle')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
    ax.axhline(90, color='grey', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=8)

    # bottom left: coding magnitude by block
    ax = axes[1, 0]
    all_mag = {'early': [], 'late': []}
    for animal in animals:
        r = results[animal]
        for block, colour in [('early', EARLY_COL), ('late', LATE_COL)]:
            ax.plot(r['t_ax'], r['magnitude'][block], color=colour,
                    alpha=0.2, linewidth=0.8)
            all_mag[block].append(r['magnitude'][block])

    for block, colour in [('early', EARLY_COL), ('late', LATE_COL)]:
        mean_mag = np.nanmean(all_mag[block], axis=0)
        ax.plot(t_ax, mean_mag, color=colour, linewidth=2, label=f'{block} block')
    ax.set_xlabel('Post-pulse time (s)')
    ax.set_ylabel('Coding vector magnitude')
    ax.set_title('TF encoding strength by block')
    ax.legend(fontsize=8)
    ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')

    # bottom right: within-block rotation rate
    ax = axes[1, 1]
    all_rot = {'early': [], 'late': []}
    for animal in animals:
        r = results[animal]
        t_rot = r['t_ax'][:-1]
        for block, colour in [('early', EARLY_COL), ('late', LATE_COL)]:
            ax.plot(t_rot, r['within_block_rotation'][block],
                    color=colour, alpha=0.2, linewidth=0.8)
            all_rot[block].append(r['within_block_rotation'][block])

    for block, colour in [('early', EARLY_COL), ('late', LATE_COL)]:
        mean_rot = np.nanmean(all_rot[block], axis=0)
        ax.plot(t_rot, mean_rot, color=colour, linewidth=2,
                label=f'{block} block')
    ax.set_xlabel('Post-pulse time (s)')
    ax.set_ylabel('Rotation rate (deg/bin)')
    ax.set_title('Within-block coding direction evolution')
    ax.legend(fontsize=8)
    ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / 'coding_rotation.png',
                    dpi=300, bbox_inches='tight')

    return fig


#%% analysis 3: motor projection heatmaps

def plot_motor_projection(npx_dir=PATHS['npx_dir_local'], save_dir=None):
    """
    plot all-to-all motor projection heatmaps + variance explained curves
    """
    with open(Path(npx_dir) / 'block_modulation' / 'motor_projection.pkl', 'rb') as f:
        results = pickle.load(f)
    animals = list(results.keys())

    # average heatmaps across animals
    heatmaps_avg = {}
    for block in ['early', 'late']:
        maps = [results[a]['heatmaps_motor'][block] for a in animals]
        heatmaps_avg[block] = np.nanmean(maps, axis=0)

    diff = heatmaps_avg['early'] - heatmaps_avg['late']
    t_tf = results[animals[0]]['tf_t_ax']
    t_lick = results[animals[0]]['lick_t_ax']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # top row: heatmaps
    vmax = max(np.abs(heatmaps_avg['early']).max(),
               np.abs(heatmaps_avg['late']).max())

    for i, (block, title) in enumerate([
            ('early', 'Early block'), ('late', 'Late block')]):
        ax = axes[0, i]
        im = ax.imshow(heatmaps_avg[block], aspect='auto',
                       origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[t_lick[0], t_lick[-1],
                               t_tf[0], t_tf[-1]])
        ax.set_xlabel('Pre-lick time (s)')
        ax.set_ylabel('Post-pulse time (s)')
        ax.set_title(title)
        ax.axhline(0, color='white', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='white', linewidth=0.5, linestyle='--')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # difference heatmap
    ax = axes[0, 2]
    vmax_diff = np.abs(diff).max()
    im = ax.imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r',
                   vmin=-vmax_diff, vmax=vmax_diff,
                   extent=[t_lick[0], t_lick[-1], t_tf[0], t_tf[-1]])
    ax.set_xlabel('Pre-lick time (s)')
    ax.set_ylabel('Post-pulse time (s)')
    ax.set_title('Difference (early - late)')
    ax.axhline(0, color='white', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='white', linewidth=0.5, linestyle='--')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # bottom left: variance explained (real vs fake lick)
    ax = axes[1, 0]
    for animal in animals:
        r = results[animal]
        n_pcs = len(r['var_explained_real'])
        ax.plot(range(1, n_pcs + 1), r['var_explained_real'],
                color='black', alpha=0.3, linewidth=0.8)
        if r['var_explained_fake'] is not None:
            ax.plot(range(1, n_pcs + 1), r['var_explained_fake'],
                    color='grey', alpha=0.3, linewidth=0.8, linestyle='--')

    # average
    mean_real = np.nanmean(
        [r['var_explained_real'] for r in results.values()], axis=0)
    ax.plot(range(1, len(mean_real) + 1), mean_real, color='black',
            linewidth=2, label='Real licks')
    fake_vals = [r['var_explained_fake'] for r in results.values()
                 if r['var_explained_fake'] is not None]
    if fake_vals:
        mean_fake = np.nanmean(fake_vals, axis=0)
        ax.plot(range(1, len(mean_fake) + 1), mean_fake, color='grey',
                linewidth=2, linestyle='--', label='Fake licks')
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Variance explained (cross-val)')
    ax.set_title('Motor subspace dimensionality')
    ax.legend(fontsize=8)

    # bottom middle: motor subspace N per animal
    ax = axes[1, 1]
    n_motors = [results[a]['n_motor'] for a in animals]
    ax.scatter(range(len(animals)), n_motors, color='grey', edgecolor='black',
               s=40, zorder=3)
    ax.set_xticks(range(len(animals)))
    ax.set_xticklabels(animals, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('N motor dimensions')
    ax.set_title('Motor subspace size per animal')

    # bottom right: total motor vs non-motor projection by block
    ax = axes[1, 2]
    motor_mag = {'early': [], 'late': []}
    nonmotor_mag = {'early': [], 'late': []}

    for animal in animals:
        r = results[animal]
        for block in ['early', 'late']:
            motor_mag[block].append(
                np.nanmean(np.abs(r['heatmaps_motor'][block])))
            nonmotor_mag[block].append(
                np.nanmean(np.abs(r['heatmaps_nonmotor'][block])))

    x = np.arange(2)
    width = 0.35
    for block, colour, offset in [('early', EARLY_COL, -width/2),
                                   ('late', LATE_COL, width/2)]:
        motor_m = np.mean(motor_mag[block])
        nonmotor_m = np.mean(nonmotor_mag[block])
        motor_s = np.std(motor_mag[block]) / np.sqrt(len(animals))
        nonmotor_s = np.std(nonmotor_mag[block]) / np.sqrt(len(animals))

        # individual animals as dots
        ax.scatter(np.zeros(len(animals)) + offset,
                   motor_mag[block], color=colour, alpha=0.5, s=20)
        ax.scatter(np.ones(len(animals)) + offset,
                   nonmotor_mag[block], color=colour, alpha=0.5, s=20)

        # mean + SEM
        ax.errorbar(x + offset, [motor_m, nonmotor_m],
                    yerr=[motor_s, nonmotor_s],
                    color=colour, linewidth=2, capsize=4,
                    label=f'{block} block', marker='o', markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(['Motor dims', 'Non-motor dims'])
    ax.set_ylabel('Mean |projection|')
    ax.set_title('Projection magnitude by dimension type')
    ax.legend(fontsize=8)

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / 'motor_projection.png',
                    dpi=300, bbox_inches='tight')

    return fig
