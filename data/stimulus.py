"""
Functions for getting stimulus types/times
"""
from config import ANALYSIS_OPTIONS
from data.session import Session

def get_tf_outliers(thresh: float = 1.0,
                    direction = 1,
                    time_range: tuple|list = [2, 8],
                    block: str = 'early',
                    ops: dict = ANALYSIS_OPTIONS):
    raise NotImplementedError

def get_baseline_onset_times(session, ops: dict = ANALYSIS_OPTIONS):


def get_change_onset_times():
    raise NotImplementedError

def get_lick_onset_times():
    raise NotImplementedError
