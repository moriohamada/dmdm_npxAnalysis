from config import PATHS, ANALYSIS_OPTIONS

#%% get all event times and event-aligned neural responses
from data.load_npx import extract_session_data
extract_session_data(npx_dir_ceph=PATHS['npx_dir_ceph'],
                     npx_dir_local=PATHS['npx_dir_local'],
                     ops=ANALYSIS_OPTIONS,
                     n_workers=4)

#%% behavioural model - predict animals' behaviour, identify important variables
from analyses.run_lick_prediction import run_lick_prediction
run_lick_prediction(npx_dir=PATHS['npx_dir_local'],
                    overwrite=True)


#%% plot single unit psths
from visualisation.psths import plot_all_su_psths
plot_all_su_psths(npx_dir=PATHS['npx_dir_local'],
                  plots_dir=PATHS['plots_dir'],
                  ops=ANALYSIS_OPTIONS,
                  n_workers=10)

#%% Extract TF preference, lick modulation by block/time
from analyses.preferences import extract_all_unit_preferences
extract_all_unit_preferences(npx_dir=PATHS['npx_dir_local'],
                             ops=ANALYSIS_OPTIONS)

#%% Visualize preferences
from visualisation.preferences import visualise_all_preferences
visualise_all_preferences(npx_dir=PATHS['npx_dir_local'],
                          save_dir=PATHS['plots_dir'],
                          ops=ANALYSIS_OPTIONS,
                          sig_flag='both',
                          alpha=.05)

#%% Fit GLMs to single units


#%% Downsample FR matrices for population analyses
from utils.downsampling import save_downsampled_fr
save_downsampled_fr(npx_dir=PATHS['npx_dir_local'],
                    ops=ANALYSIS_OPTIONS,
                    n_workers=4)

#%% PCA
from analyses.population import extract_pcs
extract_pcs(npx_dir=PATHS['npx_dir_local'],
            ops=ANALYSIS_OPTIONS)

#%% Linear dynamical systems analysis
from analyses.dynamical import run_lds_analysis
run_lds_analysis(npx_dir=PATHS['npx_dir_local'],
                 ops=ANALYSIS_OPTIONS,
                 n_workers=4)

#%% Empirical flow fields
#
# run_flow_analysis(npx_dir=PATHS['npx_dir_local'],
#                   ops=ANALYSIS_OPTIONS,
#                   n_workers=4)

#%% Visualise LDS trajectories + flow fields

from pathlib import Path
from utils.filing import get_response_files
from visualisation.dynamical import plot_session_dynamics


psth_paths = get_response_files(PATHS['npx_dir_local'])
for psth_path in psth_paths:
    sess_dir = Path(psth_path).parent
    save_dir = Path(PATHS['plots_dir']) / 'lds' / sess_dir.parent.name / sess_dir.name
    plot_session_dynamics(sess_dir,
                          pca_key='event_all',
                          ops=ANALYSIS_OPTIONS,
                          save_dir=str(save_dir))

# #%% Visualise empirical flow fields
#
# for psth_path in psth_paths:
#     sess_dir = Path(psth_path).parent
#     for event_type in ['tf', 'bl', 'lick']:
#         save_path = (Path(PATHS['plots_dir']) / 'flow' / sess_dir.parent.name
#                      / sess_dir.name / f'flow_{event_type}.png')
#         plot_empirical_flow(sess_dir,
#                             pca_key='event_all',
#                             event_type=event_type,
#                             ops=ANALYSIS_OPTIONS,
#                             save_path=str(save_path))

#%% SAE


#%%

import os
import pickle

npx_dir = '/media/morio/Data_Fast/dmdm_temporalExpectation/npx/'

all_areas = set()
for subj in os.listdir(npx_dir):
  subj_dir = os.path.join(npx_dir, subj)
  if not os.path.isdir(subj_dir):
      continue
  for sess in os.listdir(subj_dir):
      pkl_path = os.path.join(subj_dir, sess, 'session.pkl')
      if not os.path.exists(pkl_path):
          continue
      with open(pkl_path, 'rb') as f:
          s = pickle.load(f)
      if s.unit_info is not None:
          all_areas.update(s.unit_info['brain_region_comb'].values)

for a in sorted(all_areas):
  print(a)

