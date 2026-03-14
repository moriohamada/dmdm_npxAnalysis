import numpy as np
import pandas as pd

def zscore_fr(fr: np.ndarray) -> np.ndarray:
    mu = np.mean(fr, axis=1, keepdims=True)
    sigma = np.std(fr, axis=1, keepdims=True)
    zscore = np.where(sigma == 0, 0.0, (fr - mu) / sigma)
    return zscore