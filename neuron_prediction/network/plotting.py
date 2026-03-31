"""
network kernel route plots per neuron
"""
import numpy as np
import matplotlib.pyplot as plt

from neuron_prediction.glm.fit import extract_kernels
from neuron_prediction.glm.plotting import plot_glm_kernels


def plot_network_routes(w_ih, w_ho, col_map, neuron_idx=0, region=None,
                        mean_r=None, n_hidden_size=None, max_routes=8):
    """overlay per-hidden-unit kernel contributions on tf and lick panels"""
    n_hidden = w_ih.shape[0]
    nh = n_hidden_size or n_hidden

    # each route is the signed contribution of one hidden unit
    routes = w_ho[:, None] * w_ih  # (n_hidden, n_inputs)
    effective = w_ho @ w_ih        # (n_inputs,)

    # pick top routes by norm if too many
    if n_hidden > max_routes:
        norms = np.linalg.norm(routes, axis=1)
        top_idx = np.argsort(norms)[-max_routes:][::-1]
        routes = routes[top_idx]
        labels = [f'h{i}' for i in top_idx]
    else:
        labels = [f'h{i}' for i in range(n_hidden)]

    cmap = plt.colormaps['tab10']
    colours = [cmap(i % 10) for i in range(len(routes))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    avg_kernels = extract_kernels(effective, col_map)

    # individual routes
    for route, label, colour in zip(routes, labels, colours):
        kernels = extract_kernels(route, col_map)

        if 'tf' in kernels:
            t, w = kernels['tf']
            axes[0].plot(t, w, color=colour, lw=0.5, alpha=0.5, label=label)

        if 'lick_prep' in kernels:
            t, w = kernels['lick_prep']
            axes[1].plot(t, w, color=colour, lw=0.5, alpha=0.5)
        if 'lick_exec' in kernels:
            t, w = kernels['lick_exec']
            axes[1].plot(t, w, color=colour, lw=0.5, alpha=0.5)

    # average effective kernel
    if 'tf' in avg_kernels:
        t, w = avg_kernels['tf']
        axes[0].plot(t, w, 'k', lw=2, label='effective')

    if 'lick_prep' in avg_kernels:
        t, w = avg_kernels['lick_prep']
        axes[1].plot(t, w, color='steelblue', lw=2, label='prep')
    if 'lick_exec' in avg_kernels:
        t, w = avg_kernels['lick_exec']
        axes[1].plot(t, w, color='firebrick', lw=2, label='exec')

    # styling
    axes[0].set_title('Baseline TF')
    axes[0].set_xlabel('Time (s)')
    axes[1].set_title('Lick')
    axes[1].set_xlabel('Time from lick (s)')

    for ax in axes:
        ax.axhline(0, color='grey', lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    axes[1].axvline(0, color='grey', lw=0.5, ls='--')

    axes[0].legend(fontsize=6, ncol=2)
    axes[1].legend(fontsize=7)

    title = f'h={nh} - unit {neuron_idx}'
    if region:
        title += f' ({region})'
    if mean_r is not None:
        title += f' - r={mean_r:.2f}'
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    return fig
