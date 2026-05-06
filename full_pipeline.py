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


#%% Behavioural quantification
plot_dir = Path(PATHS['plots_dir']) / 'behaviour'
plot_dir.mkdir(parents=True, exist_ok=True)

from behaviour.analysis import extract_all_behavioural
extract_all_behavioural(npx_dir=PATHS['npx_dir_local'], overwrite=False)

from behaviour.analysis import load_behavioural
from behaviour.plotting import plot_psychometric, plot_elta

psycho, chrono = load_behavioural('psychometric')
plot_psychometric(psycho).write_image(str(plot_dir / 'psychometric.png'))
plot_psychometric(chrono).write_image(str(plot_dir / 'chronometric.png'))

elta = load_behavioural('elta')
plot_elta(elta).write_image(str(plot_dir / 'elta.png'))

#%% Model licks

#%% Single unit characterisation - gain/offset

