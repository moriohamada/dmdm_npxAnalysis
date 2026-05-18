"""
Full analysis pipeline for basic descriptive paper of temporal expectation.
1) Behaviour: quantify changes in gain, offset, integration.
2) Behavioural model: what causes mice to lick?
3) Single unit analyses: example psths, per-block glm fits
4) Population: ?
5) Where is prior? ITI->early baseline->mid baseline
6) Single trial trajectories
"""

from pathlib import Path
from config import *
plot_dir = Path(PATHS['plots_dir']) / 'behaviour'

#%% Behaviour visualisation
from behaviour.extraction import extract_all_behavioural, load_behavioural
from behaviour.plotting import plot_all_behavioural, plot_all_quantifications

extract_all_behavioural(npx_dir=PATHS['npx_dir_local'], overwrite=False)
plot_all_behavioural(plot_dir)

#%% Quantification of expectation-dependent behavioural changes
plot_all_quantifications(plot_dir)

#%% RNN fits to individual mice
from behaviour.extraction import load_behavioural
from behaviour_rnn.train import train_rnns_all_subj
from behaviour_rnn.plotting import comparative_plots

train_rnns_all_subj(load_behavioural('dfs_processed'))
comparative_plots(plot_dir / 'rnn')

#%% Leaky integration simulation

#%% Model licks

#%% Single unit characterisation - gain/offset/integration time
