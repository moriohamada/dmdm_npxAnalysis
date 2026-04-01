from config import PATHS, ANALYSIS_OPTIONS

#%% get all event times and event-aligned neural responses
from data.load_npx import extract_session_data
extract_session_data(npx_dir_ceph=PATHS['npx_dir_ceph'],
                     npx_dir_local=PATHS['npx_dir_local'],
                     ops=ANALYSIS_OPTIONS,
                     n_workers=4,
                     overwrite=True)


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

# Prepare design matrices
from neuron_prediction.glm.fit import prepare_all_sessions
prepare_all_sessions(overwrite=False)

from neuron_prediction.glm.fit import build_job_map
df = build_job_map()
print(f'--array=0-{len(df) - 1}')

from neuron_prediction.data import convert_job_map_to_hpc
import os
convert_job_map_to_hpc(os.path.join(PATHS['npx_dir'], 'glm_job_map.csv'))

#%% Classify units from GLM fits
from pathlib import Path
from utils.filing import get_response_files
from neuron_prediction.glm.fit import classify_units

for psth_path in get_response_files(PATHS['npx_dir_local']):
    sess_dir = Path(psth_path).parent

    if (sess_dir / 'glm_results').is_dir():
        classify_units(str(sess_dir))

# Plot GLM kernels per session
from neuron_prediction.glm.plotting import plot_all_glm_kernels

for psth_path in get_response_files(PATHS['npx_dir_local']):
    sess_dir = Path(psth_path).parent
    if (sess_dir / 'glm_results').is_dir():
        plot_all_glm_kernels(str(sess_dir),
                             plots_dir=PATHS['plots_dir'],
                             n_workers=6)

#%% fraction significant by region
from neuron_prediction.glm.plotting import plot_fraction_significant
plot_fraction_significant(npx_dir=PATHS['npx_dir_local'],
                          save_dir=PATHS['plots_dir'])

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

#%% Visualise LDS trajectories + flow fields

from pathlib import Path
from utils.filing import get_response_files
from population.plotting import plot_session_dynamics


import h5py
psth_paths = get_response_files(PATHS['npx_dir_local'])
for psth_path in psth_paths:
    sess_dir = Path(psth_path).parent
    pca_path = sess_dir / 'pca.h5'
    if not pca_path.exists():
        continue
    with h5py.File(pca_path, 'r') as f:
        pca_keys = list(f.keys())
    for pca_key in pca_keys:
        save_dir = Path(PATHS['plots_dir']) / 'lds' / sess_dir.parent.name / sess_dir.name / pca_key
        plot_session_dynamics(sess_dir,
                              pca_key=pca_key,
                              ops=ANALYSIS_OPTIONS,
                              save_dir=str(save_dir))


#%% Tuning curves
from config import PATHS, ANALYSIS_OPTIONS, TUNING_CURVE_OPS, CODING_DIM_OPS
from pathlib import Path
from tuning_curves.analysis import extract_all_tuning_curves
extract_all_tuning_curves(npx_dir=PATHS['npx_dir_local'],
                          ops=ANALYSIS_OPTIONS,
                          bm_ops=TUNING_CURVE_OPS,
                          overwrite=True)

from tuning_curves.plotting import plot_tuning_curves, plot_gain_offset_distributions
tc_plot_dir = Path(PATHS['plots_dir']) / 'tuning_curves'
plot_tuning_curves(npx_dir=PATHS['npx_dir_local'], save_dir=str(tc_plot_dir))
plot_gain_offset_distributions(npx_dir=PATHS['npx_dir_local'], save_dir=str(tc_plot_dir))

#%% coding dimensions: iterate over areas and unit filters
from config import CODING_DIM_OPS
from coding_dims.extract import extract_tf_dimensions, extract_motor_dimensions
from utils.rois import PLOT_CLASS_AREAS

areas = [None] + list(PLOT_CLASS_AREAS.keys())
unit_filters = [None, ['tf'], ['tf', 'lick_prep']]

for area in areas:
    for uf in unit_filters:
        print(f'\n=== TF dimensions: area={area}, filter={uf} ===')
        extract_tf_dimensions(npx_dir=PATHS['npx_dir_local'],
                              ops=ANALYSIS_OPTIONS,
                              bm_ops=CODING_DIM_OPS,
                              area=area,
                              unit_filter=uf,
                              n_jobs=6)

        print(f'\n=== Motor dimensions: area={area}, filter={uf} ===')
        extract_motor_dimensions(npx_dir=PATHS['npx_dir_local'],
                                 ops=ANALYSIS_OPTIONS,
                                 bm_ops=CODING_DIM_OPS,
                                 area=area,
                                 unit_filter=uf,
                                 n_jobs=6)

#%%
from coding_dims.analysis import analyse_coding_dimensions
tf_stats = analyse_coding_dimensions('tf', unit_filter=['tf'])
motor_stats = analyse_coding_dimensions('motor', unit_filter=['tf'])

#%% coding dim alignment analysis

from coding_dims.analysis import calculate_tf_motor_alignment
calculate_tf_motor_alignment(npx_dir=PATHS['npx_dir_local'],
                             bm_ops=CODING_DIM_OPS)

#%% Demixing - train and save per session
from config import DEMIXING_OPTIONS
from demixing.run import run_demixing
run_demixing(npx_dir=PATHS['npx_dir_local'],
             overwrite=True,
             ops=DEMIXING_OPTIONS)

#%% Demixing - latent PSTHs
from data.session import Session
from demixing.analysis import load_latents
from demixing.plotting import plot_latent_psths

psth_paths = get_response_files(PATHS['npx_dir_local'])
for psth_path in psth_paths:
    sess_dir = Path(psth_path).parent
    latent_path = sess_dir / f'demixing_{DEMIXING_OPTIONS["model_type"]}_latents.h5'
    if not latent_path.exists():
        continue
    latent_data = load_latents(str(sess_dir), DEMIXING_OPTIONS['model_type'])
    session = Session.load(str(sess_dir / 'session.pkl'))
    save_dir = Path(PATHS['plots_dir']) / 'demixing' / sess_dir.parent.name / sess_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_latent_psths(latent_data.z_all, latent_data, session, ops=ANALYSIS_OPTIONS)
    fig.savefig(save_dir / 'latent_psths.png', dpi=150, bbox_inches='tight')

#%% Demixing - predict licking from latents
from demixing.lick_prediction import run_latent_lick_prediction
run_latent_lick_prediction(npx_dir=PATHS['npx_dir_local'],
                           ops=DEMIXING_OPTIONS)

