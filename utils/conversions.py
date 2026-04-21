import pandas as pd
import numpy as np

def fr_to_arrays(fr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # fr dataframes (units x time) to (t_ax, fr) numpy arrays
    t_ax = fr.columns.to_numpy(dtype=float)
    return t_ax, fr.to_numpy()