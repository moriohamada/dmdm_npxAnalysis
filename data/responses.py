"""
Functions for extracting event-aligned neural responses
"""
import numpy as np
import pandas as pd
from config import PATHS, ANALYSIS_OPTIONS
from data.stimulus import *
from data.session import Session

def extract_all_timings(session: Session = None,
                        ops: dict = ANALYSIS_OPTIONS,
                        save_path: str = PATHS['npx_dir_local']
                        ):
    """
    Extract event-aligned firing rates for all units in given session.
    Saves a file with all events per event type, as well as an
     average_resp file containing mean responses for every event type.
    """

    # Baseline onsets
    session = get_baseline_onset_times(session, ops)

    # TF outliers


    # Change onset

    # Hits

    # FAs


    return session