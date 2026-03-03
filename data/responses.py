"""
Functions for extracting event-aligned neural responses
"""
import numpy as np
import pandas as pd
from config import PATHS, ANALYSIS_OPTIONS
from data.stimulus import *
from data.session import Session

def extract_all_timings(session: Session = None,
                        ops: dict = ANALYSIS_OPTIONS
                        ):
    """
    Get timings and info for task events of interest
    """
    session = get_baseline_onset_times(session)
    session = get_tf_outliers(session, ops)
    session = get_lick_onset_times(session)
    session = get_change_onset_times(session)

    return session

def get_event_aligned_responses(
        session: Session,
        ops: dict = ANALYSIS_OPTIONS,
        save_path: str = PATHS['npx_dir_local']):
    """
    Extract event-aligned firing rates for all units in given session.
    Saves a file with all events per event type, as well as an
     average_resp file containing mean responses for every event type.
    """

    return NotImplementedError

