from pathlib import Path
from config import PATHS

plot_dir = Path(PATHS['plots_dir']) / 'behaviour'
plot_dir.mkdir(parents=True, exist_ok=True)

#%% Extract and save all behavioural analyses
from behaviour.analysis import extract_all_behavioural
extract_all_behavioural(npx_dir=PATHS['npx_dir_local'], overwrite=False)

#%% Psychometric / chronometric
from behaviour.analysis import load_behavioural
from behaviour.plotting import plot_psychometric

psycho, chrono = load_behavioural('psychometric')
plot_psychometric(psycho).write_image(str(plot_dir / 'psychometric.png'))
plot_psychometric(chrono).write_image(str(plot_dir / 'chronometric.png'))

#%% Lick-triggered stimulus averages (ELTA)
from behaviour.plotting import plot_elta

elta = load_behavioural('elta')
plot_elta(elta).write_image(str(plot_dir / 'elta.png'))

#%% Lick-triggered covariance (ELTC)
from behaviour.plotting import plot_eltc, plot_eltc_comparison

eltc = load_behavioural('eltc_aligned')
plot_eltc(eltc).write_image(str(plot_dir / 'eltc.png'))
plot_eltc_comparison(eltc).write_image(str(plot_dir / 'eltc_comparison.png'))

#%% FA hazard rates
from behaviour.plotting import plot_el_hazard_rates

hazard = load_behavioural('hazard_rates')
plot_el_hazard_rates(hazard).write_image(str(plot_dir / 'hazard_rates.png'))

#%% Pulse-aligned lick probability
from behaviour.plotting import plot_pulse_aligned_lick_prob

pulse_lick = load_behavioural('pulse_lick_prob')
plot_pulse_aligned_lick_prob(pulse_lick).write_image(str(plot_dir / 'pulse_lick_prob.png'))

#%% Leaky integrator model
# from behaviour.integrator import model_integrator_by_subj
# dfs = load_behavioural('dfs_processed')
# model_integrator_by_subj(dfs, overwrite=True)

# from behaviour.model_plots import visualize_integrator_fits, visualize_best_params
# from behaviour.integrator import load_results
# params, likelihoods, lick_dists, time_bins, best_params, n_runs = load_results(...)
# visualize_integrator_fits(params, likelihoods, best_params).show()
# visualize_best_params(best_params).show()

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
