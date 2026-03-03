"""
Functions for extracting event-aligned neural responses
"""
import numpy as np
import pandas as pd
from config import PATHS, ANALYSIS_OPTIONS
from data.stimulus import *
from data.session import Session

def extract_all_responses(session: Session = None,
                          ops: dict = ANALYSIS_OPTIONS,
                          save_path: str = PATHS['npx_dir_local']
                          )-> tuple(dict, dict)
    """
    Extract event-aligned firing rates for all units in given session.
    Saves a file with all events per event type, as well as an
     average_resp file containing mean responses for every event type.
    """

    # Baseline onsets
    baseline_onsets = get_baseline_onset_times(session, ops)

    # TF outliers

    # Change onset

    # Hits

    # FAs


    return t_ax, avg_resps