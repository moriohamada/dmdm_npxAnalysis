"""
behavioural quantification: significance tests, effect sizes, model comparisons.
operates on cached results from extraction.py.
"""
from config import ANALYSIS_OPTIONS
from behaviour.extraction import load_behavioural, save_behavioural
import numpy as np
from scipy.optimize import minimize


CHANGE_TFS = np.asarray(ANALYSIS_OPTIONS['change_tfs']) - 1   # Hz above baseline; x=0 is no-change
CHANGE_GRPS = tuple('z' if tf == 0 else 's' if tf < 1 else 'l' for tf in CHANGE_TFS)   # zero/small/large change

def _fit_psychometric(n_h, n_tr, changes: list[float] = CHANGE_TFS ):
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

    returns:
        params: dict keyed by alpha, beta, gamma, lapse, log_lik, converged
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
    params = {key: np.full(n_subj, np.nan)
           for key in ('alpha', 'beta', 'gamma', 'lapse', 'log_lik')}
    params['converged'] = np.zeros(n_subj, dtype=bool)

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

        params['alpha'][animal_id] = res.x[0]
        params['beta'][animal_id] = res.x[1]
        params['gamma'][animal_id] = res.x[2]
        params['lapse'][animal_id] = res.x[3]
        params['log_lik'][animal_id] = -float(res.fun)
        params['converged'][animal_id] = bool(res.success)

    return params

def _psychometric_fit_stats(psychometric_params: dict,
                            sig_test: str = 'ttest'):
    """
    Run paired test (default t-test, or wilcoxon) on psychometric fits.
    args:
        psychometric_params: dict of fitted params from _fit_psychometric, with keys 'early' and 'late'
        sig_test: str: 'ttest' or 'wilcoxon'
    returns:
        stats: dict of {param_name: (stat, pval)}
    """

    stats = {}
    param_names = ('alpha', 'beta', 'gamma', 'lapse')

    if sig_test == 'ttest':
        from scipy.stats import ttest_1samp
        stat_fn = lambda d: ttest_1samp(d, 0)
    elif sig_test == 'wilcoxon':
        from scipy.stats import wilcoxon as stat_fn
    else:
        raise ValueError(f'unknown sig_test: {sig_test}')

    for param in param_names:
        early_vals = psychometric_params['early'][param]
        late_vals = psychometric_params['late'][param]
        stat, pval = stat_fn(early_vals - late_vals)
        stats[param] = (stat, pval)

    return stats


def _quantify_hit_change(valsE, valsL, nE, nL,
                         changes=CHANGE_TFS, change_grps=CHANGE_GRPS,
                         sig_test='ttest'):
    """
    pairwise stats on hit rates by change-group (z/s/l) and block (E/L):
        E vs L within each group
        s vs z, l vs z, l vs s within each block
        (s-z), (l-z), (l-s) differences compared between blocks (across-animal only)

    per animal: fisher's exact on trial counts for each comparison.
    across animals: paired ttest (default) or wilcoxon on per-animal rates.

    args:
        valsE, valsL: (n_subj, n_chs) hit rates per block
        nE, nL: (n_subj, n_chs) trial counts per block
        changes: (n_chs,) Hz above baseline, kept for api consistency
        change_grps: (n_chs,) 'z'/'s'/'l' label per change TF
        sig_test: 'ttest' or 'wilcoxon' for across-animal tests
    returns:
        subj_stats: {subj: {comparison: (delta, pval)}}
        group_stats: {comparison: (delta, pval)}
    """
    from scipy.stats import fisher_exact, ttest_rel, wilcoxon

    if sig_test == 'ttest':
        paired_test = ttest_rel
    elif sig_test == 'wilcoxon':
        paired_test = wilcoxon
    else:
        raise ValueError(f'unknown sig_test: {sig_test}')

    change_grps = np.asarray(change_grps)
    grp_cols = {g: np.where(change_grps == g)[0]
                for g in sorted(set(change_grps))}

    def aggregate(vals, n, cols):
        """sum hits and trials over a set of change-TF columns, per animal"""
        h_raw = np.where(n[:, cols] > 0, vals[:, cols] * n[:, cols], 0)
        return h_raw.sum(axis=1), n[:, cols].sum(axis=1)

    grp_data = {}
    for g, cols in grp_cols.items():
        grp_data[g] = {
            'E': aggregate(valsE, nE, cols),
            'L': aggregate(valsL, nL, cols),
        }

    def grp_rate(g, block):
        hits, trials = grp_data[g][block]
        return np.where(trials > 0, hits / np.maximum(trials, 1), np.nan)

    n_subj = valsE.shape[0]
    subj_stats = {i: {} for i in range(n_subj)}
    group_stats = {}

    def compare(hits1, trials1, hits2, trials2, label):
        """fisher per animal, paired test across animals; writes both outputs"""
        rate1 = np.where(trials1 > 0, hits1 / np.maximum(trials1, 1), np.nan)
        rate2 = np.where(trials2 > 0, hits2 / np.maximum(trials2, 1), np.nan)
        for i in range(n_subj):
            table = [[int(hits1[i]), int(trials1[i] - hits1[i])],
                     [int(hits2[i]), int(trials2[i] - hits2[i])]]
            _, p = fisher_exact(table)
            subj_stats[i][label] = (float(rate1[i] - rate2[i]), float(p))
        _, p = paired_test(rate1, rate2, nan_policy='omit')
        group_stats[label] = (float(np.nanmean(rate1 - rate2)), float(p))

    # E vs L within each group
    for g in grp_cols:
        (hE, tE), (hL, tL) = grp_data[g]['E'], grp_data[g]['L']
        compare(hE, tE, hL, tL, f'{g}_E-L')

    # pair comparisons within each block
    pairs = [('s', 'z'), ('l', 'z'), ('l', 's')]
    for g1, g2 in pairs:
        for block in ('E', 'L'):
            h1, t1 = grp_data[g1][block]
            h2, t2 = grp_data[g2][block]
            compare(h1, t1, h2, t2, f'{g1}-{g2}_{block}')

    # difference-of-differences (across-animal only): (g1-g2)_E vs (g1-g2)_L
    for g1, g2 in pairs:
        d_E = grp_rate(g1, 'E') - grp_rate(g2, 'E')
        d_L = grp_rate(g1, 'L') - grp_rate(g2, 'L')
        _, p = paired_test(d_E, d_L, nan_policy='omit')
        group_stats[f'{g1}-{g2}_E-L'] = (
            float(np.nanmean(d_E - d_L)), float(p))

    return subj_stats, group_stats


def _quantify_rt_change(valsE, valsL, nE, nL,
                        changes=CHANGE_TFS, change_grps=CHANGE_GRPS,
                        sig_test='ttest'):
    """
    across-animal paired stats on RTs by change-group and block.
    same comparisons as _quantify_hit_change. no per-animal test
    (would need trial-level RTs; only cell means are available).

    aggregates within each group as trial-weighted mean RT across change-TFs.

    args:
        valsE, valsL: (n_subj, n_chs) mean RT per cell
        nE, nL: (n_subj, n_chs) trial counts (for weighting + empty-cell mask)
        sig_test: 'ttest' or 'wilcoxon'

    returns:
        group_stats: {comparison: (delta, pval)}
    """
    from scipy.stats import ttest_rel, wilcoxon

    if sig_test == 'ttest':
        paired_test = ttest_rel
    elif sig_test == 'wilcoxon':
        paired_test = wilcoxon
    else:
        raise ValueError(f'unknown sig_test: {sig_test}')

    change_grps = np.asarray(change_grps)
    grp_cols = {g: np.where(change_grps == g)[0]
                for g in sorted(set(change_grps))}

    def weighted_mean(vals, n, cols):
        v, w = vals[:, cols], n[:, cols]
        mask = ~np.isnan(v) & (w > 0)
        v_safe = np.where(mask, v, 0)
        w_safe = np.where(mask, w, 0)
        num = (v_safe * w_safe).sum(axis=1)
        den = w_safe.sum(axis=1)
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)

    grp_rt = {g: {'E': weighted_mean(valsE, nE, cols),
                  'L': weighted_mean(valsL, nL, cols)}
              for g, cols in grp_cols.items()}

    group_stats = {}

    def compare(a, b, label):
        _, p = paired_test(a, b, nan_policy='omit')
        group_stats[label] = (float(np.nanmean(a - b)), float(p))

    for g in grp_cols:
        compare(grp_rt[g]['E'], grp_rt[g]['L'], f'{g}_E-L')

    pairs = [('s', 'z'), ('l', 'z'), ('l', 's')]
    for g1, g2 in pairs:
        for block in ('E', 'L'):
            compare(grp_rt[g1][block], grp_rt[g2][block], f'{g1}-{g2}_{block}')

    for g1, g2 in pairs:
        d_E = grp_rt[g1]['E'] - grp_rt[g2]['E']
        d_L = grp_rt[g1]['L'] - grp_rt[g2]['L']
        compare(d_E, d_L, f'{g1}-{g2}_E-L')

    return group_stats

def quantify_change_detection(psycho: np.ndarray,
                              chrono: np.ndarray,
                              n_hits: np.ndarray,
                              n_trials: np.ndarray,
                              config=ANALYSIS_OPTIONS):
    """
    shape of psycho/chrono/n_h/n_trials are all:
     animals x change_TF x block (early/late) x probe (expected/unexpected timing)

    Test difference between psychometric fits, hit rates/rt to compare
    early vs late blocks, early in trial.
    """

    e_block = np.s_[:, :, 0, 0]  # early block, non-probe
    l_block = np.s_[:, :, 1, 1]  # late block, probe

    psychometric_params = {
        'early': _fit_psychometric(n_hits[e_block], n_trials[e_block], CHANGE_TFS),
        'late':  _fit_psychometric(n_hits[l_block], n_trials[l_block], CHANGE_TFS),
    }
    psychometric_param_stats = _psychometric_fit_stats(psychometric_params,
                                                       sig_test='ttest')

    # compute delta and signf directly on change hits/rt dists
    _, hit_stats_grp = _quantify_hit_change(
        psycho[e_block], psycho[l_block],
        n_trials[e_block], n_trials[l_block],
        CHANGE_TFS, CHANGE_GRPS)
    rt_stats_grp = _quantify_rt_change(
        chrono[e_block], chrono[l_block],
        n_trials[e_block], n_trials[l_block],
        CHANGE_TFS, CHANGE_GRPS)

    return psychometric_params, psychometric_param_stats, hit_stats_grp, rt_stats_grp


def quantify_lick_triggered_stim(elts: dict, config=ANALYSIS_OPTIONS):
    """
    1) Quantify difference in mean AND variance between lick-triggered stimuli
    2) Extract common pc dims for early and late, find diffrences in
    occupancy/projections of stimuli along each to detect possible strategy change or
    component-specifc changes
    """


def quantify_pulse_lick_probability():

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


