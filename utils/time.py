"""time window helpers"""

import numpy as np


def window_label(win):
    return f'{win[0]:.2f}_{win[1]:.2f}'


def time_mask(t_ax, win):
    """mask for time bins within window [win[0], win[1])"""
    return (t_ax >= win[0]) & (t_ax < win[1])
