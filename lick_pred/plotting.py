"""
plotting for lick prediction model results
"""
import numpy as np
import matplotlib.pyplot as plt
from lick_pred.features import FEATURE_COLS


def plot_calibration(y_true, y_pred, n_bins=10, ax=None):
    """predicted P(lick) vs observed lick rate"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    observed = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() > 0:
            observed[i] = y_true[mask].mean()

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.plot(bin_centres, observed, 'o-')
    ax.set_xlabel('Predicted P(lick)')
    ax.set_ylabel('Observed lick rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def plot_ablation(ablation_results, ax=None):
    """bar plot of mean loss increase per feature group"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    names = list(ablation_results.keys())
    means = [np.nanmean(ablation_results[n]) for n in names]
    sems = [np.nanstd(ablation_results[n]) / np.sqrt(np.sum(~np.isnan(ablation_results[n])))
            for n in names]

    order = np.argsort(means)[::-1]
    ax.barh(range(len(names)), [means[i] for i in order],
            xerr=[sems[i] for i in order], capsize=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel('Loss increase (ablation)')
    return ax


def plot_stimulus_filter(weights, bin_width=0.05, ax=None):
    """stimulus history weights from the linear model"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))

    t = np.arange(len(weights)) * bin_width - len(weights) * bin_width
    ax.plot(t, weights)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Time before current bin (s)')
    ax.set_ylabel('Weight')
    return ax


def plot_temporal_accuracy(lick_times_true, peak_times_pred, ax=None):
    """histogram of predicted peak time - actual lick time"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))

    errors = np.array(peak_times_pred) - np.array(lick_times_true)
    ax.hist(errors, bins=50, edgecolor='k', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Peak prediction - actual lick (s)')
    ax.set_ylabel('Count')
    return ax


def plot_model_comparison(linear_losses, network_losses, ax=None):
    """scatter of linear vs network test loss per session"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(linear_losses, network_losses, alpha=0.7)
    lims = [min(min(linear_losses), min(network_losses)),
            max(max(linear_losses), max(network_losses))]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('Linear test loss')
    ax.set_ylabel('Network test loss')
    return ax
