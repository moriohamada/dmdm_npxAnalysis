import numpy as np
import pandas as pd


def causal_boxcar(data, window_bins):
    """
    Causal (past-only) boxcar smooth along time axis (axis=1).

    Args:
        data:        2D numpy array or pandas DataFrame (n_neurons x n_bins)
        window_bins: number of bins to average over (including current bin)

    Returns:
        Smoothed array/DataFrame in same format as input
    """
    is_df = isinstance(data, pd.DataFrame)
    arr = data.values if is_df else np.asarray(data)
    window_bins = int(round(window_bins))
    assert arr.ndim == 2, "Input must be 2D"

    kernel = np.ones(window_bins) / window_bins
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode='full')[:arr.shape[1]],
        axis=1,
        arr=arr
    )

    if is_df:
        return pd.DataFrame(smoothed, index=data.index, columns=data.columns)
    return smoothed


def causal_gaussian(data, sigma_bins, truncate=3.0):
    """
    Causal (past-only) Gaussian smooth along time axis (axis=1).
    """
    is_df = isinstance(data, pd.DataFrame)
    arr = data.values if is_df else np.asarray(data)

    assert arr.ndim == 2, "Input must be 2D (n_neurons x n_bins)"

    # Build one-sided Gaussian kernel
    half_width = int(truncate * sigma_bins)
    x = np.arange(0, half_width + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()  # normalise

    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode='full')[:arr.shape[1]],
        axis=1,
        arr=arr
    )

    if is_df:
        return pd.DataFrame(smoothed, index=data.index, columns=data.columns)
    return smoothed