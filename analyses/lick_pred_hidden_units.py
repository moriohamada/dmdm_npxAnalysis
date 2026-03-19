"""
hidden unit analysis for lick prediction network models
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from config import PATHS
from data.lick_features import FEATURE_COLS, CONTINUOUS_COLS, N_TF_HIST, OUTCOME_MAP
from analyses.lick_prediction import NetworkLickModel, evaluate, compute_class_weight
from analyses.lick_pred_model_analyses import (
    load_results, load_mouse, predict_session, BIN_WIDTH, DRIVER_GROUPS,
    _reconstruct_model, _non_stimulus_features,
)


#%% compute

def analyse_hidden_units(mouse, all_res, arch):
    """ablate each hidden unit per fold to test effects, and extract input weights

    for each fold: zero out one hidden unit at a time, measure loss increase
    also extracts first-layer input weights for interpreting what units respond to

    returns dict with:
        importance: (n_folds, n_hidden) loss increase per unit
        input_weights: (n_hidden, n_features) first-layer weights (from fold 0)
        input_bias: (n_hidden,) first-layer biases (from fold 0)
        output_weights: (n_hidden,) second-layer weights (from fold 0)
    """
    raise NotImplementedError


#%% plots

def plot_unit_summary(unit_result, animal, arch, top_n=5, save_path=None):
    """input weight profiles and output weight for the top-N units in one mouse

    units ranked by mean importance across folds
    per unit row: stimulus filter (left), non-stimulus weights (middle), output weight (right)
    """
    raise NotImplementedError


def plot_unit_responses(mouse, all_res, arch, unit_results, sess_idx,
                        top_n=3, n_trials=5, save_path=None):
    """hidden unit activations over time for top units in example trials

    overlaid with lick target to see when each unit fires relative to licking
    """
    raise NotImplementedError


def plot_unit_trial_type(mouse, all_res, arch, unit_result, top_n=5,
                         save_path=None):
    """loss increase per hidden unit, split by trial type, for one mouse

    shows whether ablating a unit disproportionately hurts hit vs FA vs miss predictions
    """
    raise NotImplementedError


#%% run
all_res = load_results()
mice = {a: load_mouse(a, all_res) for a in all_res}
arch = 'h8'
unit_results = {a: analyse_hidden_units(mice[a], all_res, arch) for a in mice}

#%% unit importance + weights (per mouse)
for animal in mice:
    plot_unit_summary(unit_results[animal], animal, arch, top_n=5)

#%% unit response profiles
animal = list(mice.keys())[0]
plot_unit_responses(mice[animal], all_res, arch, unit_results[animal], sess_idx=0)

#%% unit-by-trial-type (per mouse)
for animal in mice:
    plot_unit_trial_type(mice[animal], all_res, arch, unit_results[animal])
