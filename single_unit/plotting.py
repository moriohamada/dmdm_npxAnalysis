import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd
import numpy as np
from pathlib import Path
from config import ANALYSIS_OPTIONS, PATHS
from utils.filing import get_response_files
from single_unit.preferences import (combine_preference_data, clean_preference_data)




def scatter_preference_indexes(prefs: pd.DataFrame,
                               plot_pair: tuple[str,str],
                               sig_flag: str | None = None,
                               alpha: float = .05,
                               ):
    """
    Scatter plot of preference indexes
    - prefs: dataframe of nN by preference columns (_idx, _p)
    - plot_pair: tuples of preference types to plot
    - sig_flag: None - plot all neurons; 'either' - plot only units with significant
    preference in either index; 'both' - plot only units with both indexes significant
    """
    fig, ax = plt.subplots()
    if sig_flag is None:
        mask = np.ones(len(prefs), dtype=bool)
    elif sig_flag == 'either':
        mask = (prefs[plot_pair[0] + '_p']<alpha) | (prefs[plot_pair[1] + '_p']<alpha)
    elif sig_flag == 'both':
        mask = (prefs[plot_pair[0] + '_p']<alpha) & (prefs[plot_pair[1] + '_p']<alpha)
    else:
        raise Exception(f"Invalid value for sig_flag: {sig_flag}")

    sns.scatterplot(data=prefs.loc[mask,:],
                    x=plot_pair[0]+"_idx",
                    y=plot_pair[1]+"_idx",
                    ax=ax)
    ax.set_xlim((-1,1))
    ax.set_ylim((-1,1))
    ax.set_xlabel(plot_pair[0])
    ax.set_ylabel(plot_pair[1])
    return fig

def visualise_all_preferences(npx_dir: str = PATHS['npx_dir_local'],
                              save_dir: str = PATHS['plots_dir'],
                              ops: dict = ANALYSIS_OPTIONS,
                              sig_flag: str = 'both',
                              alpha: float = .05):
    """
    Visualize preferences of all units across all sessions:
    1) TF, early vs late block
    2) lick, early vs late block
    3) block preference vs TF/lick
    """
    # collect data across all recording sessions
    psth_paths = get_response_files(npx_dir)
    preference_paths = [path.replace('psths.h5', 'preferences.csv') for path in psth_paths]
    prefs = combine_preference_data(preference_paths)
    prefs = clean_preference_data(prefs)

    # iterate through pairings to plat
    plot_pairs = dict(
        block_dependent_tf = ('tf_earlyBlock_early', 'tf_lateBlock_early'),
        time_dependent_tf = ('tf_lateBlock_early', 'tf_lateBlock_late'),
        tf_time = ('tf', 'time'),
        tf_lick = ('tf', 'lick'),
        block_depdent_lick = ('lick_earlyBlock_early', 'lick_lateBlock_early')
    )

    plot_dir = Path(save_dir) / 'preferences'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for name, pp in plot_pairs.items():
        fig = scatter_preference_indexes(prefs, plot_pair = pp, sig_flag=sig_flag,
                                         alpha=alpha)
        fig.savefig(plot_dir / f'{name}.png')
        plt.close(fig)



