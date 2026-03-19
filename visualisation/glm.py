"""
glm kernel plots per neuron
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

from config import GLM_OPTIONS, PLOT_OPTIONS
from analyses.glm import extract_kernels


def plot_glm_kernels(weights, col_map, neuron_idx=0, region=None,
                     mean_r=None, classifications=None,
                     save_dir=None):
    """plot all GLM kernels for one neuron"""
    kernels = extract_kernels(weights, col_map)

    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    axes = axes.ravel()

    # panel 0: baseline TF
    ax = axes[0]
    if 'tf' in kernels:
        t, w = kernels['tf']
        ax.plot(t, w, 'k', lw=1.5)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Baseline TF')
    ax.set_xlabel('Time (s)')

    # panel 1: change onset (overlaid by magnitude)
    ax = axes[1]
    ch_keys = sorted([k for k in kernels if k.startswith('change_tf')])
    ch_cmap = plt.cm.get_cmap(PLOT_OPTIONS['colours']['ch_tf_cmap'])
    if ch_keys:
        colours = ch_cmap(np.linspace(0.15, 0.85, len(ch_keys)))
        colours[0] = (0.6, 0.6, 0.6, 1.0)
        for key, colour in zip(ch_keys, colours):
            t, w = kernels[key]
            label = key.replace('change_tf', '')
            ax.plot(t, w, color=colour, lw=1.2, label=label)
        ax.legend(fontsize=6, title='TF', title_fontsize=6)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Change onset')
    ax.set_xlabel('Time (s)')

    # panel 2: lick prep + exec
    ax = axes[2]
    if 'lick_prep' in kernels:
        t, w = kernels['lick_prep']
        ax.plot(t, w, color='steelblue', lw=1.5, label='prep')
    if 'lick_exec' in kernels:
        t, w = kernels['lick_exec']
        ax.plot(t, w, color='firebrick', lw=1.5, label='exec')
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_title('Lick')
    ax.set_xlabel('Time from lick (s)')
    ax.legend(fontsize=7)

    # panel 3: trial start
    ax = axes[3]
    if 'trial_start' in kernels:
        t, w = kernels['trial_start']
        ax.plot(t, w, 'k', lw=1.5)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Trial start')
    ax.set_xlabel('Time (s)')

    # panel 4: air-puff, reward, abort
    ax = axes[4]
    event_preds = [('air_puff', 'orange', 'air-puff'),
                   ('reward', 'green', 'reward'),
                   ('abort', 'red', 'abort')]
    for name, colour, label in event_preds:
        if name in kernels:
            t, w = kernels[name]
            ax.plot(t, w, color=colour, lw=1.2, label=label)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Air-puff / reward / abort')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=7)

    # panel 5: face motion energy
    ax = axes[5]
    if 'face_me' in kernels:
        t, w = kernels['face_me']
        ax.plot(t, w, 'k', lw=1.5)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Face motion energy')
    ax.set_xlabel('Time (s)')

    # panel 6: running
    ax = axes[6]
    if 'running' in kernels:
        t, w = kernels['running']
        ax.plot(t, w, 'k', lw=1.5)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Running wheel')
    ax.set_xlabel('Time (s)')

    # panel 7: pupil
    ax = axes[7]
    if 'pupil' in kernels:
        t, w = kernels['pupil']
        ax.plot(t, w, 'k', lw=1.5)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Pupil')
    ax.set_xlabel('Time (s)')

    # title
    title = f'Unit {neuron_idx}'
    if region:
        title += f' ({region})'
    if mean_r is not None:
        title += f' - r={mean_r:.2f}'
    if classifications:
        flags = [k.replace('_sig', '') for k, v in classifications.items()
                 if k.endswith('_sig') and v]
        if flags:
            title += f' [{", ".join(flags)}]'
    fig.suptitle(title, fontsize=11)

    for a in axes:
        a.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / f'glm_unit_{neuron_idx:04d}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_all_glm_kernels(sess_dir, plots_dir=None):
    """plot GLM kernels for all neurons in one session"""
    import pickle
    from data.session import Session

    sess_dir = Path(sess_dir)
    sess = Session.load(str(sess_dir / 'session.pkl'))
    results_dir = sess_dir / 'glm_results'

    with open(sess_dir / 'glm_spec.pkl', 'rb') as f:
        col_map = pickle.load(f)

    # load classifications if available
    class_path = sess_dir / 'glm_classifications.csv'
    class_df = None
    if class_path.exists():
        class_df = pd.read_csv(class_path)

    if plots_dir is None:
        plots_dir = sess_dir / 'glm_kernels'
    save_dir = Path(plots_dir) / sess.animal / sess.name / 'glm_kernels'

    regions = sess.unit_info['brain_region_comb'].values
    n_neurons = len(sess.fr_stats)

    for i in range(n_neurons):
        res_path = results_dir / f'neuron_{i}.npz'
        if not res_path.exists():
            continue

        res = np.load(res_path, allow_pickle=True)
        weights = res['weights']

        mean_r = np.nanmean(res['full_r'])
        classifications = None
        if class_df is not None and i < len(class_df):
            row = class_df.iloc[i]
            classifications = {c: row[c] for c in row.index if c.endswith('_sig')}

        plot_glm_kernels(weights, col_map,
                         neuron_idx=i,
                         region=regions[i] if i < len(regions) else None,
                         mean_r=mean_r,
                         classifications=classifications,
                         save_dir=str(save_dir))

    print(f'Saved kernel plots to {save_dir}')
