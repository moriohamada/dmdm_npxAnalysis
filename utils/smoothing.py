import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.ndimage import uniform_filter1d


def causal_boxcar(data, window_bins, axis=-1):
    is_df = isinstance(data, pd.DataFrame)
    arr = data.values if is_df else np.asarray(data)
    window_bins = int(round(window_bins))

    kernel = np.ones(window_bins) / window_bins
    smoothed = lfilter(kernel, 1.0, arr, axis=axis)

    # fix edge bins: rescale to account for incomplete window
    for i in range(window_bins - 1):
        slc = [slice(None)] * arr.ndim
        slc[axis] = i
        smoothed[tuple(slc)] *= window_bins / (i + 1)

    if is_df:
        return pd.DataFrame(smoothed, index=data.index, columns=data.columns)
    return smoothed

def centred_boxcar(data, window_bins, axis=-1):
    """centred (non-causal) boxcar smooth, nan-aware"""
    arr = np.asarray(data, dtype=float)
    window_bins = int(round(window_bins))
    if window_bins <= 1:
        return arr.copy()
    # ensure odd window for symmetric centering
    if window_bins % 2 == 0:
        window_bins += 1
    valid = ~np.isnan(arr)
    arr_filled = np.where(valid, arr, 0.0)
    sums = uniform_filter1d(arr_filled, size=window_bins, axis=axis,
                            mode='constant', cval=0.0)
    counts = uniform_filter1d(valid.astype(float), size=window_bins, axis=axis,
                              mode='constant', cval=0.0)
    with np.errstate(invalid='ignore'):
        result = sums / counts
    result[counts == 0] = np.nan
    return result


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