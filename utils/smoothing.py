import numpy as np
import pandas as pd
from scipy.signal import lfilter

def causal_boxcar(data, window_bins, axis=-1):
    is_df = isinstance(data, pd.DataFrame)
    arr = data.values if is_df else np.asarray(data)
    window_bins = int(round(window_bins))

    kernel = np.ones(window_bins) / window_bins
    smoothed = lfilter(kernel, 1.0, arr, axis=axis)

    # replace edge bins with original values
    edge = [slice(None)] * arr.ndim
    edge[axis] = slice(0, window_bins - 1)
    smoothed[tuple(edge)] = arr[tuple(edge)]

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