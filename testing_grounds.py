from config import PATHS, ANALYSIS_OPTIONS

#%% get all event times and event-aligned neural responses
from data.load_npx import extract_session_data
extract_session_data(npx_dir_ceph=PATHS['npx_dir_ceph'],
                     npx_dir_local=PATHS['npx_dir_local'],
                     ops=ANALYSIS_OPTIONS,
                     n_workers=4,
                     overwrite=True)

#%% behavioural model - predict animals' behaviour, identify important variables
from lick_pred.run import run_lick_prediction
# run_lick_prediction(npx_dir=PATHS['npx_dir_local'],
#                     overwrite=True)

# # local parallelized:
# from config import LICK_PRED_OPS
# from lick_pred.run import _group_sessions_by_mouse, run_single_mouse
# from multiprocessing import Pool
#
# ops = {**LICK_PRED_OPS, 'max_epochs': 250, 'patience': 15}
# save_dir = '/tmp/lick_pred_test'
#
# grouped = _group_sessions_by_mouse(PATHS['npx_dir_local'])
# def _run(args):
#     animal, paths = args
#     run_single_mouse(animal, paths, save_dir, ops=ops)
# with Pool(4) as pool:
#     pool.map(_run, sorted(grouped.items()))

from lick_pred.analysis import run_all_lick_model_analyses
run_all_lick_model_analyses()


#%% plot single unit psths
from single_unit.psths import plot_all_su_psths
plot_all_su_psths(npx_dir=PATHS['npx_dir_local'],
                  plots_dir=PATHS['plots_dir'],
                  ops=ANALYSIS_OPTIONS,
                  n_workers=10)

#%% Extract TF preference, lick modulation by block/time
from single_unit.preferences import extract_all_unit_preferences
extract_all_unit_preferences(npx_dir=PATHS['npx_dir_local'],
                             ops=ANALYSIS_OPTIONS)

#%% Visualize preferences
from single_unit.plotting import visualise_all_preferences
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
from population.pca import extract_pcs
extract_pcs(npx_dir=PATHS['npx_dir_local'],
            ops=ANALYSIS_OPTIONS)

#%% PC 'psths'
from population.plotting import plot_all_pc_psths
plot_all_pc_psths(npx_dir=PATHS['npx_dir_local'],
                  plots_dir=PATHS['plots_dir'],
                  ops=ANALYSIS_OPTIONS)

#%% Linear dynamical systems analysis
from population.dynamical import run_lds_analysis
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
from population.plotting import plot_session_dynamics


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

#%% Demixing (SAE / LFADS)
import numpy as np
from config import DEMIXING_OPTIONS
from data.session import Session
from utils.filing import load_fr_matrix
from demixing.dataset import SpikeData
from demixing.models import SAE
from demixing.train import train as train_demixing
from demixing.analysis import extract_latents
from demixing.plotting import plot_latent_psths

sess_dir = Path(PATHS['npx_dir_local']) / '1116764' / 'ML_1116764_S01_V1'
session = Session.load(str(sess_dir / 'session.pkl'))
fr = load_fr_matrix(str(sess_dir / 'FR_matrix.parquet'))

dataset = SpikeData(session, fr)
n_neurons = dataset.X.shape[0]
n_latent = int(np.round(n_neurons * DEMIXING_OPTIONS['latent_factor']))
model = SAE(n_neurons=n_neurons, latent_dim=n_latent)

losses = train_demixing(dataset, model, ops=DEMIXING_OPTIONS)

#%%
z = extract_latents(dataset, model)
fig = plot_latent_psths(z, dataset, session, ops=ANALYSIS_OPTIONS)

