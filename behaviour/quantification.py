"""
behavioural quantification: significance tests, effect sizes, model comparisons.
operates on cached results from extraction.py.
"""
from config import ANALYSIS_OPTIONS
from behaviour.extraction import load_behavioural, save_behavioural
import numpy as np
from scipy.optimize import minimize


CHANGE_TFS = np.asarray(ANALYSIS_OPTIONS['change_tfs']) - 1   # Hz above baseline; x=0 is no-change
CHANGE_GRPS = ['z' if tf==0 else 's' if tf<1 else 'l' for tf in CHANGE_TFS] # zero/small/large change

def _fit_psychometric(n_h, n_tr, changes: list[float] = CHANGE_TFS):
    """
    Fit 4-param Weibull psychometric per animal w binomial MLE:

        P(hit | x) = gamma + (1 - gamma - lapse) * (1 - exp(-(x / alpha)**beta))

    parameters:
        gamma:  FA rate at no change
        lapse:  inattention floor below 1; ceiling = 1 - lapse
        alpha:  threshold scale in Hz above baseline; the x at which the curve is ~63%
                between gamma and (1 - lapse) - horizontal pos
        beta:   slope shape/steepness.
                beta=1 -> exponential rise (log-like).
                beta>1 -> more sigmoidal
                beta<1 -> even more saturating

    n_h, n_tr shape (n_subj, n_chs); returns dict of (n_subj,) arrays
    keyed by alpha, beta, gamma, lapse, log_lik, converged.
    """

    def p_hit(alpha, beta, gamma, lapse, x):
        # avoid div-by-zero on tiny alpha; ratio clip stops overflow at large x/alpha
        ratio = np.clip(x / max(alpha, 1e-9), 0, 50) ** beta
        return gamma + (1 - gamma - lapse) * (1 - np.exp(-ratio))

    def nll(params):
        alpha, beta, gamma, lapse = params
        p = np.clip(p_hit(alpha, beta, gamma, lapse, tfs), 1e-9, 1 - 1e-9)
        return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum()

    n_subj = n_h.shape[0]
    out = {key: np.full(n_subj, np.nan)
           for key in ('alpha', 'beta', 'gamma', 'lapse', 'log_lik')}
    out['converged'] = np.zeros(n_subj, dtype=bool)

    changes = np.asarray(changes)

    for animal_id in range(n_subj):
        valid = n_tr[animal_id] > 0
        if valid.sum() < 4:
            continue

        tfs = changes[valid]
        k = n_h[animal_id, valid].astype(float)
        n = n_tr[animal_id, valid].astype(float)

        # initial guesses: alpha ~ median non-zero change, beta=1 (log-like),
        # gamma from x=0 row, lapse from top two bins
        emp_rate = np.where(n > 0, k / n, 0.5)
        gamma0 = float(emp_rate[0]) if tfs[0] == 0 else 0.05
        lapse0 = float(1 - emp_rate[-2:].mean())
        nonzero = tfs[tfs > 0]
        alpha0 = float(np.median(nonzero)) if len(nonzero) else 1.0
        x0 = [max(alpha0, 1e-2),
              1.0,
              np.clip(gamma0, 0.01, 0.5),
              np.clip(lapse0, 0.01, 0.5)]
        bounds = [(1e-3, float(tfs.max()) * 5),
                  (0, 10.0),
                  (0.0, 0.5),
                  (0.0, 0.5)]
        try:
            res = minimize(nll, x0, bounds=bounds, method='L-BFGS-B')
        except Exception as e:
            print(f'animal {animal_id} fit failed: {type(e).__name__}: {e}')
            continue

        out['alpha'][animal_id] = res.x[0]
        out['beta'][animal_id] = res.x[1]
        out['gamma'][animal_id] = res.x[2]
        out['lapse'][animal_id] = res.x[3]
        out['log_lik'][animal_id] = -float(res.fun)
        out['converged'][animal_id] = bool(res.success)

    return out


def _quantify_perf_change(valsE, valsL, changes, change_labels):
    """
    singned rank/permutation test on differences in hit rates or rts; also hit:fa ratios
    """
    pass

def quantify_change_detection(psycho: np.ndarray,
                              chrono: np.ndarray,
                              n_hits: np.ndarray,
                              n_trials: np.ndarray,
                              config=ANALYSIS_OPTIONS):
    """
    shape of psycho/chrono/n_h/n_trials are all:
     animals x change_TF x block (early/late) x probe (expected/unexpected timing)

    Test difference between psychometric fits, and rmANOVA for hit rates/rt to compare
    early vs late blocks, early in trial.
    """

    e_block = np.s_[:, :, 0, 0]  # early block, non-probe
    l_block = np.s_[:, :, 1, 1]  # late block, probe

    psychometric_params = {
        'early': _fit_psychometric(n_hits[e_block], n_trials[e_block], CHANGE_TFS),
        'late':  _fit_psychometric(n_hits[l_block], n_trials[l_block], CHANGE_TFS),
    }

    # compute delta and signf by change size (zero/small/large - anova?)
    psycho_stats =  _quantify_perf_change(psycho[e_block], psycho[l_block],
                                          CHANGE_TFS, CHANGE_GRPS)


    chrono_stats = {'early': {},
                    'late': {}}








def quantify_pulse_lick_probability():
    pass

def quantify_lick_triggered_stim():
    pass

def quantify_hazard_rates():
    pass

def quantify_integration_time():
    pass

def run_all_quantifications(config=ANALYSIS_OPTIONS, overwrite=False):

    cached = load_behavioural('stats') if not overwrite else None
    if cached is not None:
        return cached

    psycho, chrono, n_hits, n_trials = load_behavioural('psychometric')
    elts = load_behavioural('elts')
    hazard = load_behavioural('hazard_rates')
    pulse_lick = load_behavioural('pulse_lick_prob')

    stats = {
        'psychometric': quantify_change_detection(psycho, chrono, n_hits, n_trials, config),
        'lts': quantify_lick_triggered_stim(elts, config),
        'hazard_rates': quantify_hazard_rates(hazard, config),
        'pulse_lick_prob': quantify_pulse_lick_probability(pulse_lick, config),
        'integration_time': quantify_integration_time(pulse_lick, config),
    }
    save_behavioural(stats, 'stats')
    return stats


