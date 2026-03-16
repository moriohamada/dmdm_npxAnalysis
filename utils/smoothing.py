import numpy as np
import pandas as pd
from scipy.signal import lfilter


def downsample_bins(data, factor, axis=-1):
    """
    Downsample by averaging groups of `factor` adjacent bins along axis.
    Truncates any remainder bins at the end.
    Works on ndarrays and DataFrames (downsamples columns for DataFrames).
    """
    factor = int(round(factor))
    if factor <= 1:
        return data

    is_df = isinstance(data, pd.DataFrame)
    if is_df:
        arr = data.values
        cols = data.columns.values
        n_keep = (len(cols) // factor) * factor
        new_cols = cols[:n_keep].reshape(-1, factor).mean(axis=1)
        new_vals = arr[:, :n_keep].reshape(arr.shape[0], -1, factor).mean(axis=2)
        return pd.DataFrame(new_vals, index=data.index, columns=new_cols)

    arr = np.asarray(data)
    n = arr.shape[axis]
    n_keep = (n // factor) * factor
    slc = [slice(None)] * arr.ndim
    slc[axis] = slice(0, n_keep)
    trimmed = arr[tuple(slc)]
    new_shape = list(trimmed.shape)
    new_shape[axis] = n_keep // factor
    new_shape.insert(axis + 1, factor)
    return trimmed.reshape(new_shape).mean(axis=axis + 1)

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

    # Build one-sided gaussian kernel
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