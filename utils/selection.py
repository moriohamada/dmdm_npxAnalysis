"""
Functions for selecting/filtering units and trials.
"""
import numpy as np
import pandas as pd


def filter_units(fr_stats: pd.DataFrame,
                 min_fr: float = 1.0,
                 min_fr_sd: float = 0.5) -> np.ndarray:
    """
    Return boolean mask (nN,) for units passing FR criteria based on fr mean and sd
    """
    return ((fr_stats['mean'] >= min_fr) &
            (fr_stats['sd'] >= min_fr_sd)).values
