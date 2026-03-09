"""
Functions for plotting event-aligned PSTHS
"""
import os
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from config import ANALYSIS_OPTIONS, PATHS
from data.session import Session
from analyses.load_responses import load_psth

from utils.filing import get_response_files

def plot_psth(t_ax: np.ndarray,
              mu: np.ndarray,
              err: np.ndarray,
              ):
    raise NotImplementedError

def plot_raster(t_ax: np.ndarray,
                arr: np.ndarray):
    raise NotImplementedError

def plot_basic_psths(psth_path: str,
                     save_dir: str = None):
    """
    Plot psths for all units in a session:
    1) baseline onsets (col 1)
    2) TF pulses: E/L block, E/L in trial (col 2 fast. col 3 slow)
    3) changes: split by magnitude, E/L block )
    4) FAs: E/L block, E/L in trial
    """
    ncol = 7
    fig = plt.figure()

    # baseline onset
    bl_conds =
    axbl_psth = fig.add_subplot(2, ncol, 1)
    ax_bl_raster = fig.add_subplot(2, ncol, 2)


    # tf pulses

    # changes

    # FAs






def plot_all_su_psths(npx_dir: str = PATHS['npx_dir_local'],
                      ops=ANALYSIS_OPTIONS):
    """
    Runs through all units to generate basic single unit PSTHs
    """
    psth_paths = get_response_files(npx_dir)

    for psth_path in psth_paths:
        # make basic psth/raster visual
        plot_basic_psth(psth_path)



