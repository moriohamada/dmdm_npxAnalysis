"""
behavioural quantification: significance tests, effect sizes, model comparisons.
operates on cached results from extraction.py.
"""
from config import ANALYSIS_OPTIONS
from behaviour.extraction import load_behavioural, save_behavioural
import numpy as np

def quantify_psychometrics(psycho: np.ndarray,
                           chrono: np.ndarray,
                           config=ANALYSIS_OPTIONS):
    """
    shape: animals x change_TF x block (early/late) x probe (expected/unexpected timing)
    """

    pass

def quantify_pulse_lick_probability():
    pass

def quantify_lick_triggered_stim():
    pass

def quantify_hazard_rates():
    pass

def quantify_integration_time():
    pass

def run_all_quantifications(config=ANALYSIS_OPTIONS, overwrite=False):

    cached = load_behavioural('stats') if not overwrite else None
    if cached is not None:
        return cached

    psycho, chrono = load_behavioural('psychometric')
    elts = load_behavioural('elts')
    hazard = load_behavioural('hazard_rates')
    pulse_lick = load_behavioural('pulse_lick_prob')

    stats = {
        'psychometric': quantify_psychometrics(psycho, chrono, config),
        'lts': quantify_lick_triggered_stim(elts, config),
        'hazard_rates': quantify_hazard_rates(hazard, config),
        'pulse_lick_prob': quantify_pulse_lick_probability(pulse_lick, config),
        'integration_time': quantify_integration_time(pulse_lick, config),
    }
    save_behavioural(stats, 'stats')
    return stats


