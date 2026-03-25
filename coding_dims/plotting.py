"""plotting for coding dimension rotation and motor projection analyses"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

sns.set_style("whitegrid")

from config import PATHS, PLOT_OPTIONS


EARLY_COL = PLOT_OPTIONS['colours']['block']['early']
LATE_COL = PLOT_OPTIONS['colours']['block']['late']
FAST_COL = PLOT_OPTIONS['colours']['tf_pref']['fast']
SLOW_COL = PLOT_OPTIONS['colours']['tf_pref']['slow']


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
