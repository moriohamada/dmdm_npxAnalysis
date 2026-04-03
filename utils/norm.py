import numpy as np
import pandas as pd

def zscore_fr(fr: np.ndarray) -> np.ndarray:
    mu = np.mean(fr, axis=1, keepdims=True)
    sigma = np.std(fr, axis=1, keepdims=True)
    zscore = np.where(sigma == 0, 0.0, (fr - mu) / sigma)
    return zscore


def baseline_subtract(trace: np.ndarray, t_ax: np.ndarray,
                      bl_window: tuple[float, float]) -> np.ndarray:
    """subtract mean activity in a baseline time window from a trace"""
    mask = (t_ax >= bl_window[0]) & (t_ax < bl_window[1])
    return trace - np.nanmean(trace[mask])