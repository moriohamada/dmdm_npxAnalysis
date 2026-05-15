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
from utils.figures import save_fig


#%% Behaviour visualization
plot_dir = Path(PATHS['plots_dir']) / 'behaviour'
plot_dir.mkdir(parents=True, exist_ok=True)

from behaviour.extraction import extract_all_behavioural
extract_all_behavioural(npx_dir=PATHS['npx_dir_local'], overwrite=True)

from behaviour.extraction import load_behavioural
from behaviour.plotting import (plot_psychometric, plot_elta, plot_eltc,
    plot_eltc_comparison, plot_el_hazard_rates, plot_pulse_aligned_lick_prob,
    plot_pulse_lick_prob_2d)


psycho, chrono, n_hits, n_trials = load_behavioural('psychometric')
save_fig(plot_psychometric(psycho), str(plot_dir / 'psychometric'))
save_fig(plot_psychometric(chrono), str(plot_dir / 'chronometric'))

elta = load_behavioural('elta')
save_fig(plot_elta(elta), str(plot_dir / 'elta'))

eltc = load_behavioural('eltc_aligned')
save_fig(plot_eltc(eltc), str(plot_dir / 'eltc'))
save_fig(plot_eltc_comparison(eltc), str(plot_dir / 'eltc_comparison'))

hazard = load_behavioural('hazard_rates')
save_fig(plot_el_hazard_rates(hazard), str(plot_dir / 'hazard_rates'))

pulse_lick = load_behavioural('pulse_lick_prob')
for label, fig in plot_pulse_aligned_lick_prob(pulse_lick).items():
    save_fig(fig, str(plot_dir / f'pulse_lick_prob_{label}'))
for label, fig in plot_pulse_lick_prob_2d(pulse_lick).items():
    save_fig(fig, str(plot_dir / f'pulse_lick_prob_2d_{label}'))

from behaviour.two_pulse_analyses import (plot_two_pulse_interaction,
    plot_two_pulse_raw)
two_pulse = load_behavioural('two_pulse_interaction')
save_fig(plot_two_pulse_interaction(two_pulse), str(plot_dir / 'two_pulse_interaction'))
save_fig(plot_two_pulse_raw(two_pulse), str(plot_dir / 'two_pulse_raw'))

#%% Quantification of expectation-dependent behavioural changes

from behaviour.quantification import run_all_quantifications
from behaviour.plotting import (plot_psychometric_fits, plot_lts_quant,
    plot_lts_pcs, plot_hazard_rate_stats, plot_pulse_lick_fits,
    plot_integration_time)

stats = run_all_quantifications(overwrite=False)

# psychometric fits per block
psycho_params, _, _, _ = stats['psychometric']
save_fig(plot_psychometric_fits(psycho_params, n_hits, n_trials),
         str(plot_dir / 'psychometric_fits'))

# lick-triggered stim quantification
save_fig(plot_lts_quant(stats['lts']), str(plot_dir / 'lts_quant'))
save_fig(plot_lts_pcs(stats['lts']), str(plot_dir / 'lts_pcs'))

# hazard rate stats
save_fig(plot_hazard_rate_stats(stats['hazard_rates']),
         str(plot_dir / 'hazard_rate_stats'))

# pulse-aligned lick prob OLS fits per time window
save_fig(plot_pulse_lick_fits(stats['pulse_lick_prob']),
         str(plot_dir / 'pulse_lick_fits'))

# behavioural integration time from two-pulse interaction
save_fig(plot_integration_time(stats['integration_time']),
         str(plot_dir / 'integration_time'))

#%% Leaky integration simulation



#%% Model licks


#%% Single unit characterisation - gain/offset/integration time

# visualize activity profiles (tf tuning curves)

# per-block glm results

