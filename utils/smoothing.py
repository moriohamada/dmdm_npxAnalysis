import numpy as np
import pandas as pd
from scipy.signal import lfilter


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