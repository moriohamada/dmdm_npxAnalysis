"""plotting for single-unit TF tuning curve analysis"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

sns.set_style("whitegrid")

from config import PATHS, ANALYSIS_OPTIONS, PLOT_OPTIONS
from data.load_responses import load_psth_mean
from utils.filing import get_response_files
from utils.rois import AREA_GROUPS, in_group
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
    valid_x = x[~np.isnan(x)]
    margin = (valid_x[-1] - valid_x[0]) * 0.05
    x_lim = (valid_x[0] - margin, valid_x[-1] + margin)
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


            ax.set_xlim(x_lim)
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
    session_areas = []
    session_meta = []

    for result in all_results:
        session_binned.append(result['binned'])
        session_bin_centres.append(result['bin_centres'])
        gp = result['gain_p_block']
        session_sig.append((gp[:, 0] < 0.025) | (gp[:, 1] < 0.025))
        session_gains.append(result['gain_pooled'])
        ui = result['unit_info']
        session_areas.append(ui['brain_region_comb'].values
                             if hasattr(ui, 'brain_region_comb') else
                             np.array(['?'] * result['binned']['early'].shape[1]))
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

    # collect areas across all animals (same order as grand)
    grand_areas = np.concatenate([
        np.concatenate([session_areas[i] for i in animals[a]])
        for a in animal_list])

    sp = save_dir / 'tuning_curves_grand.png' if save_dir else None
    fig = _make_tuning_figure(grand, bin_centres, grand_groups, grand_sig,
                               grand_gains, 'Grand average', sp)
    plt.close(fig)

    # --- per area group ---
    for group_name in AREA_GROUPS:
        area_mask = in_group(grand_areas, group_name)
        if area_mask.sum() < 5:
            continue
        area_binned = {block: grand[block][:, area_mask] for block in ['early', 'late']}
        area_sig = grand_sig[area_mask]
        area_gains = grand_gains[area_mask]

        # group indices: one group per animal (units from that animal in this area)
        area_groups = []
        offset = 0
        for a in animal_list:
            animal_mask = in_group(
                np.concatenate([session_areas[i] for i in animals[a]]),
                group_name)
            n = animal_mask.sum()
            if n > 0:
                area_groups.append(np.arange(offset, offset + n))
            offset += n

        sp = (save_dir / f'tuning_curves_{group_name}.png'
              if save_dir else None)
        fig = _make_tuning_figure(area_binned, bin_centres, area_groups,
                                   area_sig, area_gains,
                                   f'{group_name} (n={area_mask.sum()})', sp)
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
    valid_x = bin_centres[~np.isnan(bin_centres)]
    margin = (valid_x[-1] - valid_x[0]) * 0.05
    ax.set_xlim(valid_x[0] - margin, valid_x[-1] + margin)
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
    all_gain_p_block = []
    all_gain_pooled = []

    for result in all_results:
        all_gain.append(result['gain'])
        all_offset.append(result['offset'])
        all_gain_diff_p.append(result['gain_diff_p'])
        all_offset_diff_p.append(result['offset_diff_p'])
        all_gain_p_block.append(result['gain_p_block'])
        all_gain_pooled.append(result['gain_pooled'])

    gain = np.concatenate(all_gain)            # (nN_total, 2)
    offset = np.concatenate(all_offset)
    gain_diff_p = np.concatenate(all_gain_diff_p)
    offset_p = np.concatenate(all_offset_diff_p)
    gain_p_block = np.concatenate(all_gain_p_block)  # (nN_total, 2)
    gain_pooled = np.concatenate(all_gain_pooled)
    gain_diff = gain[:, 0] - gain[:, 1]
    offset_diff = offset[:, 0] - offset[:, 1]

    # all-units masks (block diff significance)
    sig = gain_diff_p < 0.05
    nonsig = ~sig
    fast_sig = sig & (gain_pooled > 0)
    slow_sig = sig & (gain_pooled < 0)

    # TF-responsive: significant gain in either block (.025 - roughly gives .05 FWER)
    tf_resp = (gain_p_block[:, 0] < 0.025) | (gain_p_block[:, 1] < 0.025)
    fast_resp = tf_resp & (gain_pooled > 0)
    slow_resp = tf_resp & (gain_pooled < 0)
    n_tf = tf_resp.sum()
    n_fast = fast_resp.sum()
    n_slow = slow_resp.sum()

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

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

    #%% row 1: all units

    # gain scatter (all units, coloured by sig gain diff)
    ax = axes[0, 0]
    ax.scatter(gain[nonsig, 0], gain[nonsig, 1], s=8, alpha=0.3,
               color='grey', label=f'n.s. gain diff (n={nonsig.sum()})', zorder=1)
    if fast_sig.any():
        ax.scatter(gain[fast_sig, 0], gain[fast_sig, 1], s=12, alpha=0.5,
                   color=FAST_COL, label=f'Sig gain diff, fast (n={fast_sig.sum()})', zorder=3)
    if slow_sig.any():
        ax.scatter(gain[slow_sig, 0], gain[slow_sig, 1], s=12, alpha=0.5,
                   color=SLOW_COL, label=f'Sig gain diff, slow (n={slow_sig.sum()})', zorder=3)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color='black', linewidth=1, linestyle='--', zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Gain (early block)')
    ax.set_ylabel('Gain (late block)')
    ax.set_title(f'All units: gain (n={len(gain)})')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper left')

    # offset scatter (all units, coloured by sig offset diff)
    ax = axes[0, 1]
    sig_off = offset_p < 0.05
    nonsig_off = ~sig_off
    ax.scatter(offset[nonsig_off, 0], offset[nonsig_off, 1], s=8, alpha=0.3,
               color='grey', label=f'n.s. (n={nonsig_off.sum()})', zorder=1)
    if sig_off.any():
        ax.scatter(offset[sig_off, 0], offset[sig_off, 1], s=12, alpha=0.5,
                   color=EARLY_COL, label=f'Sig offset diff (n={sig_off.sum()})', zorder=3)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color='black', linewidth=1, linestyle='--', zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Offset (early block)')
    ax.set_ylabel('Offset (late block)')
    ax.set_title(f'All units: offset (n={len(offset)})')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper left')

    # gain diff histogram (all units)
    ax = axes[0, 2]
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
    ax.set_title('All units: gain diff')
    ax.legend(fontsize=8)

    # offset diff histogram (all units)
    ax = axes[0, 3]
    ax.hist(offset_diff, bins=50, color='grey', alpha=0.7, edgecolor='black',
            linewidth=0.5)
    if sig_off.any():
        ax.hist(offset_diff[sig_off], bins=50, color=EARLY_COL, alpha=0.7,
                label=f'p < 0.05 (n={sig_off.sum()})')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(np.nanmedian(offset_diff), color='red', linewidth=1.5,
               label=f'median = {np.nanmedian(offset_diff):.4f}')
    ax.set_xlabel('Offset difference (early - late)')
    ax.set_ylabel('Number of units')
    ax.set_title('All units: offset diff')
    ax.legend(fontsize=8)

    #%% row 2: TF-responsive units only

    # gain scatter (TF-responsive, by pref)
    ax = axes[1, 0]
    _scatter_early_late(ax, gain[tf_resp], fast_resp[tf_resp], slow_resp[tf_resp],
                        'Gain (early block)', 'Gain (late block)',
                        f'TF-responsive: gain (n={n_tf})')

    # offset scatter (TF-responsive, by pref)
    ax = axes[1, 1]
    _scatter_early_late(ax, offset[tf_resp], fast_resp[tf_resp], slow_resp[tf_resp],
                        'Offset (early block)', 'Offset (late block)',
                        f'TF-responsive: offset (n={n_tf})')

    # gain diff by pref (TF-responsive)
    ax = axes[1, 2]
    if n_fast > 0:
        ax.hist(gain_diff[fast_resp], bins=30, color=FAST_COL, alpha=0.5,
                label=f'Fast (n={n_fast})')
    if n_slow > 0:
        ax.hist(gain_diff[slow_resp], bins=30, color=SLOW_COL, alpha=0.5,
                label=f'Slow (n={n_slow})')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Gain difference (early - late)')
    ax.set_ylabel('Number of units')
    ax.set_title('TF-responsive: gain diff by pref')
    ax.legend(fontsize=7)

    # offset diff by pref (TF-responsive)
    ax = axes[1, 3]
    if n_fast > 0:
        ax.hist(offset_diff[fast_resp], bins=30, color=FAST_COL, alpha=0.5,
                label=f'Fast (n={n_fast})')
    if n_slow > 0:
        ax.hist(offset_diff[slow_resp], bins=30, color=SLOW_COL, alpha=0.5,
                label=f'Slow (n={n_slow})')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Offset difference (early - late)')
    ax.set_ylabel('Number of units')
    ax.set_title('TF-responsive: offset diff by pref')
    ax.legend(fontsize=7)

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / 'gain_offset_distributions.png',
                    dpi=300, bbox_inches='tight')

    return fig
