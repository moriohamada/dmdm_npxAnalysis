"""
visualisation for latent representations from demixing models
"""
import numpy as np
import matplotlib.pyplot as plt
from config import ANALYSIS_OPTIONS
from demixing.analysis import (
    align_to_baseline_onset, align_to_change_onset,
    align_to_tf_outliers, align_to_lick_onset,
)


def plot_latent_psths(z_all, dataset, session, ops=ANALYSIS_OPTIONS):
    """quick overview: trial onset, TF outliers, change onset, lick onset"""
    ign = ops['ignore_first_trials_in_block']
    rmv = ops['rmv_time_around']

    onset, t_onset = align_to_baseline_onset(z_all, dataset, session,
                                              pre_s=1.0, post_s=2.0,
                                              tr_in_block_min=ign)
    tf_dict, t_tf = align_to_tf_outliers(z_all, dataset, session,
                                          pre_s=0.5, post_s=1.5,
                                          tr_in_block_min=ign,
                                          rmv_near_response=rmv,
                                          tr_time_range=(1.5, 8))
    change, t_ch = align_to_change_onset(z_all, dataset, session,
                                          pre_s=0.5, post_s=1.5,
                                          tr_in_block_min=ign)
    lick, t_lick = align_to_lick_onset(z_all, dataset, session,
                                        pre_s=1.5, post_s=0.5,
                                        tr_in_block_min=ign)

    n_latent = z_all[0].shape[1]
    fig, axes = plt.subplots(n_latent, 4, figsize=(14, 1.8 * n_latent),
                             sharex='col', squeeze=False)
    titles = ['Trial onset', 'TF outliers', 'Change onset', 'Lick onset']

    for lat in range(n_latent):
        ax = axes[lat, 0]
        if onset is not None:
            _shade(ax, t_onset, onset[:, :, lat], 'k')
        ax.axvline(0, color='grey', ls='--', lw=0.8)
        ax.set_ylabel(f'z{lat}')

        ax = axes[lat, 1]
        ax.axvline(0, color='grey', ls='--', lw=0.8)
        if tf_dict['fast'] is not None:
            _shade(ax, t_tf, tf_dict['fast'][:, :, lat], 'green', 'fast')
        if tf_dict['slow'] is not None:
            _shade(ax, t_tf, tf_dict['slow'][:, :, lat], 'red', 'slow')
        if lat == 0:
            ax.legend(fontsize=7)

        ax = axes[lat, 2]
        ax.axvline(0, color='grey', ls='--', lw=0.8)
        if change is not None:
            _shade(ax, t_ch, change[:, :, lat], 'steelblue')

        ax = axes[lat, 3]
        ax.axvline(0, color='grey', ls='--', lw=0.8)
        if lick is not None:
            _shade(ax, t_lick, lick[:, :, lat], 'coral')

    for col, title in enumerate(titles):
        axes[0, col].set_title(title)

    _cleanup(fig, axes)
    return fig


def plot_baseline_psths(z_all, dataset, session, ops=ANALYSIS_OPTIONS,
                        pre_s=1.0, post_s=2.0):
    """baseline onset PSTHs, early vs late block"""
    ign = ops['ignore_first_trials_in_block']

    early, t = align_to_baseline_onset(z_all, dataset, session,
                                        pre_s=pre_s, post_s=post_s,
                                        block='early', tr_in_block_min=ign)
    late, _ = align_to_baseline_onset(z_all, dataset, session,
                                       pre_s=pre_s, post_s=post_s,
                                       block='late', tr_in_block_min=ign)

    n_latent = z_all[0].shape[1]
    fig, axes = plt.subplots(n_latent, 1, figsize=(5, 1.8 * n_latent),
                             sharex=True, squeeze=False)
    for lat in range(n_latent):
        ax = axes[lat, 0]
        if early is not None:
            _shade(ax, t, early[:, :, lat], 'dodgerblue', 'early')
        if late is not None:
            _shade(ax, t, late[:, :, lat], 'orangered', 'late')
        ax.axvline(0, color='grey', ls='--', lw=0.8)
        ax.set_ylabel(f'z{lat}')
        if lat == 0:
            ax.set_title('Baseline onset')
            ax.legend(fontsize=7)

    axes[-1, 0].set_xlabel('Time from baseline onset (s)')
    _cleanup(fig, axes)
    return fig


def plot_tf_psths(z_all, dataset, session, ops=ANALYSIS_OPTIONS,
                  pre_s=0.5, post_s=1.5):
    """TF outlier PSTHs: 3 columns (earlyBlock_early, lateBlock_early, lateBlock_late)"""
    ign = ops['ignore_first_trials_in_block']
    rmv = ops['rmv_time_around']
    split = ops['tr_split_time']

    conditions = {
        'Early block\n(early trial)': dict(block='early', tr_time_range=(rmv, split)),
        'Late block\n(early trial)':  dict(block='late',  tr_time_range=(rmv, split)),
        'Late block\n(late trial)':   dict(block='late',  tr_time_range=(split, np.inf)),
    }

    n_latent = z_all[0].shape[1]
    n_cols = len(conditions)
    fig, axes = plt.subplots(n_latent, n_cols,
                             figsize=(4 * n_cols, 1.8 * n_latent),
                             sharex=True, sharey='row', squeeze=False)

    for col, (title, kw) in enumerate(conditions.items()):
        tf_dict, t = align_to_tf_outliers(
            z_all, dataset, session,
            pre_s=pre_s, post_s=post_s,
            tr_in_block_min=ign, rmv_near_response=rmv, **kw)

        for lat in range(n_latent):
            ax = axes[lat, col]
            ax.axvline(0, color='grey', ls='--', lw=0.8)
            if tf_dict['fast'] is not None:
                _shade(ax, t, tf_dict['fast'][:, :, lat], 'green', 'fast')
            if tf_dict['slow'] is not None:
                _shade(ax, t, tf_dict['slow'][:, :, lat], 'red', 'slow')
            if lat == 0:
                ax.set_title(title)
                ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel(f'z{lat}')

    axes[-1, n_cols // 2].set_xlabel('Time from TF outlier (s)')
    _cleanup(fig, axes)
    return fig


def plot_change_psths(z_all, dataset, session, ops=ANALYSIS_OPTIONS,
                      pre_s=0.5, post_s=1.5):
    """change onset PSTHs: columns = block, hit vs miss overlay"""
    ign = ops['ignore_first_trials_in_block']

    n_latent = z_all[0].shape[1]
    fig, axes = plt.subplots(n_latent, 2, figsize=(8, 1.8 * n_latent),
                             sharex=True, sharey='row', squeeze=False)

    for col, block in enumerate(['early', 'late']):
        hit, t = align_to_change_onset(z_all, dataset, session,
                                        pre_s=pre_s, post_s=post_s,
                                        block=block, tr_in_block_min=ign,
                                        is_hit=1)
        miss, _ = align_to_change_onset(z_all, dataset, session,
                                         pre_s=pre_s, post_s=post_s,
                                         block=block, tr_in_block_min=ign,
                                         is_hit=0)
        for lat in range(n_latent):
            ax = axes[lat, col]
            ax.axvline(0, color='grey', ls='--', lw=0.8)
            if hit is not None:
                _shade(ax, t, hit[:, :, lat], 'steelblue', 'hit')
            if miss is not None:
                _shade(ax, t, miss[:, :, lat], 'grey', 'miss')
            if lat == 0:
                ax.set_title(f'{block.capitalize()} block')
                ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel(f'z{lat}')

    axes[-1, 0].set_xlabel('Time from change onset (s)')
    _cleanup(fig, axes)
    return fig


def plot_change_psths_by_tf(z_all, dataset, session, ops=ANALYSIS_OPTIONS,
                            pre_s=0.5, post_s=1.5):
    """change onset PSTHs split by change TF, hits only"""
    ign = ops['ignore_first_trials_in_block']
    ch_tfs = np.sort(session.ch_onsets['ch_tf'].unique())

    n_latent = z_all[0].shape[1]
    n_cols = len(ch_tfs)
    fig, axes = plt.subplots(n_latent, n_cols,
                             figsize=(3.5 * n_cols, 1.8 * n_latent),
                             sharex=True, sharey='row', squeeze=False)

    for col, ch_tf in enumerate(ch_tfs):
        for block, color in [('early', 'dodgerblue'), ('late', 'orangered')]:
            data, t = align_to_change_onset(
                z_all, dataset, session,
                pre_s=pre_s, post_s=post_s,
                block=block, tr_in_block_min=ign,
                is_hit=1, ch_tf=ch_tf)
            for lat in range(n_latent):
                ax = axes[lat, col]
                if data is not None:
                    _shade(ax, t, data[:, :, lat], color, block)

        for lat in range(n_latent):
            ax = axes[lat, col]
            ax.axvline(0, color='grey', ls='--', lw=0.8)
            if lat == 0:
                ax.set_title(f'TF = {ch_tf}')
                ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel(f'z{lat}')

    axes[-1, n_cols // 2].set_xlabel('Time from change onset (s)')
    _cleanup(fig, axes)
    return fig


def plot_lick_psths(z_all, dataset, session, ops=ANALYSIS_OPTIONS,
                    pre_s=1.5, post_s=0.5):
    """lick onset PSTHs: 3 columns, hit vs FA overlay"""
    ign = ops['ignore_first_trials_in_block']
    rmv = ops['rmv_time_around']
    split = ops['tr_split_time']

    conditions = {
        'Early block\n(early trial)': dict(block='early', tr_time_range=(rmv, split)),
        'Late block\n(early trial)':  dict(block='late',  tr_time_range=(rmv, split)),
        'Late block\n(late trial)':   dict(block='late',  tr_time_range=(split, np.inf)),
    }

    n_latent = z_all[0].shape[1]
    n_cols = len(conditions)
    fig, axes = plt.subplots(n_latent, n_cols,
                             figsize=(4 * n_cols, 1.8 * n_latent),
                             sharex=True, sharey='row', squeeze=False)

    for col, (title, kw) in enumerate(conditions.items()):
        hit, t = align_to_lick_onset(z_all, dataset, session,
                                      pre_s=pre_s, post_s=post_s,
                                      tr_in_block_min=ign, is_hit=1, **kw)
        fa, _ = align_to_lick_onset(z_all, dataset, session,
                                     pre_s=pre_s, post_s=post_s,
                                     tr_in_block_min=ign, is_FA=1, **kw)
        for lat in range(n_latent):
            ax = axes[lat, col]
            ax.axvline(0, color='grey', ls='--', lw=0.8)
            if hit is not None:
                _shade(ax, t, hit[:, :, lat], 'steelblue', 'hit')
            if fa is not None:
                _shade(ax, t, fa[:, :, lat], 'salmon', 'FA')
            if lat == 0:
                ax.set_title(title)
                ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel(f'z{lat}')

    axes[-1, n_cols // 2].set_xlabel('Time from lick onset (s)')
    _cleanup(fig, axes)
    return fig


#%% helpers

def _shade(ax, t, data, color, label=None):
    mean = data.mean(axis=0)
    sem = data.std(axis=0) / np.sqrt(data.shape[0])
    ax.plot(t, mean, color=color, lw=1.2, label=label)
    ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.2)


def _cleanup(fig, axes):
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
