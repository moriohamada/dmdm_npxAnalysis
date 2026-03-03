"""
Functions for extracting event-aligned neural responses
"""
import numpy as np
import pandas as pd
from config import PATHS, ANALYSIS_OPTIONS


def extract_all_responses(fr_matrix: pd.DataFrame,
                          trials: pd.DataFrame,
                          daq: pd.DataFrame,
                          move: dict,
                          save_path: str = PATHS['npx_dir_local']):
    raise NotImplementedError