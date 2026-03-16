from data.load_npx import extract_session_data
from visualisation.psths import plot_all_su_psths
from visualisation.preferences import visualise_all_preferences
from visualisation.dynamical import plot_session_dynamics
from analyses.preferences import extract_all_unit_preferences
from analyses.population import extract_pcs
from analyses.dynamical import run_lds_analysis
from config import PATHS, ANALYSIS_OPTIONS

#%% get all event times and event-aligned responses

extract_session_data(npx_dir_ceph=PATHS['npx_dir_ceph'],
                     npx_dir_local=PATHS['npx_dir_local'],
                     ops=ANALYSIS_OPTIONS,
                     n_workers=4)

#%% plot single unit psths

plot_all_su_psths(npx_dir=PATHS['npx_dir_local'],
                  plots_dir=PATHS['plots_dir'],
                  ops=ANALYSIS_OPTIONS,
                  n_workers=10)

#%% Extract TF preference, lick modulation by block/time

extract_all_unit_preferences(npx_dir=PATHS['npx_dir_local'],
                             ops=ANALYSIS_OPTIONS)

#%% Visualize preferences

visualise_all_preferences(npx_dir=PATHS['npx_dir_local'],
                          save_dir=PATHS['plots_dir'],
                          ops=ANALYSIS_OPTIONS,
                          sig_flag='both',
                          alpha=.05)

#%% Fit GLMs to single units


#%% PCA

extract_pcs(npx_dir=PATHS['npx_dir_local'],
            ops=ANALYSIS_OPTIONS)

#%% Dynamical systems analysis

run_lds_analysis(npx_dir=PATHS['npx_dir_local'],
                 ops=ANALYSIS_OPTIONS,
                 pca_key='event_all')

#%% Visualise LDS flow fields + trajectories

from pathlib import Path
from utils.filing import get_response_files
from analyses.dynamical import CONDITIONS

psth_paths = get_response_files(PATHS['npx_dir_local'])
for psth_path in psth_paths[:1]:
    sess_dir = Path(psth_path).parent
    for lds_cond in CONDITIONS:
        save_path = (Path(PATHS['plots_dir']) / 'LDS' / sess_dir.parent.name
                     / sess_dir.name / f'lds_{lds_cond}.png')
        plot_session_dynamics(sess_dir,
                              pca_key='event_frontal_motor',
                              lds_cond=lds_cond,
                              ops=ANALYSIS_OPTIONS,
                              save_path=str(save_path))

#%% SAE
