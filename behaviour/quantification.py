"""
behavioural quantification: significance tests, effect sizes, model comparisons.
operates on cached results from extraction.py.
"""
import os
from config import ANALYSIS_OPTIONS
from behaviour.extraction import (load_behavioural, save_behavioural,
                                  BEHAVIOUR_DATA_DIR)
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ttest_rel, wilcoxon
from sklearn.decomposition import PCA

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


def quantify_lick_triggered_stim(elts: dict, config=ANALYSIS_OPTIONS,
                                 n_components=3, sig_test='ttest', seed=0):
    """
    quantify earlyBlock_early vs lateBlock_early lick-triggered stims:
    1) paired test per timepoint on per-animal ELTA (mean) and ELTV (variance)
    2) single global PCA on pooled stims across all animals and all 3 lts
       conditions, with per-animal counts matched across conditions (each
       condition subsampled to the smallest of the animal's 3). Each animal
       then has its earlyBlock_early and lateBlock_early stims projected
       onto the shared basis; paired test per component on projection mean
       and variance.

    args:
        elts: dict from extract_elts; uses earlyBlock_early, lateBlock_early,
              lateBlock_late
        n_components: number of common PCs to summarise
        sig_test: 'ttest' or 'wilcoxon'
        seed: rng seed for per-animal subsampling

    returns dict with:
        subjs                                          list of animals
        elta_E, elta_L, eltv_E, eltv_L                 (n_subj, n_samples)
        mean_stats, var_stats                          (n_samples, 2) stat, pval per timepoint
        pc_components                                  (n_components, n_samples)
        pc_explained_var                               (n_components,)
        projection_mean_E, projection_mean_L           (n_subj, n_components)
        projection_var_E, projection_var_L             (n_subj, n_components)
        projection_mean_stats, projection_var_stats    list of (stat, pval) per component
    """


    if sig_test == 'ttest':
        paired_test = ttest_rel
    elif sig_test == 'wilcoxon':
        paired_test = wilcoxon
    else:
        raise ValueError(f'unknown sig_test: {sig_test}')

    rel_idx = np.s_[:, 10:]
    animals = elts['earlyBlock_early'].keys()
    rng = np.random.default_rng(seed)
    conds = ('earlyBlock_early', 'lateBlock_early', 'lateBlock_late')

    subjs = []
    elta_E, elta_L, eltv_E, eltv_L = [], [], [], []
    pca_pool = []

    for animal in animals:
        per_cond = [elts[c][animal][rel_idx] for c in conds]
        if any(len(x) < 2 for x in per_cond):
            continue
        e, l, _ = per_cond

        elta_E.append(e.mean(axis=0))
        elta_L.append(l.mean(axis=0))
        eltv_E.append(e.var(axis=0))
        eltv_L.append(l.var(axis=0))

        # match counts across this animal's 3 conditions before adding to global PCA pool
        n_match = min(len(x) for x in per_cond)
        for x in per_cond:
            pca_pool.append(x[rng.choice(len(x), n_match, replace=False)])

        subjs.append(animal)

    elta_E = np.stack(elta_E)
    elta_L = np.stack(elta_L)
    eltv_E = np.stack(eltv_E)
    eltv_L = np.stack(eltv_L)

    pca = PCA(n_components=n_components).fit(np.vstack(pca_pool))
    pc_components = pca.components_
    pc_explained = pca.explained_variance_ratio_

    projection_mean_E, projection_mean_L = [], []
    projection_var_E, projection_var_L = [], []
    for animal in subjs:
        e = elts['earlyBlock_early'][animal][rel_idx]
        l = elts['lateBlock_early'][animal][rel_idx]
        e_scores = (e - pca.mean_) @ pc_components.T
        l_scores = (l - pca.mean_) @ pc_components.T
        projection_mean_E.append(e_scores.mean(axis=0))
        projection_mean_L.append(l_scores.mean(axis=0))
        projection_var_E.append(e_scores.var(axis=0))
        projection_var_L.append(l_scores.var(axis=0))

    projection_mean_E = np.stack(projection_mean_E)
    projection_mean_L = np.stack(projection_mean_L)
    projection_var_E = np.stack(projection_var_E)
    projection_var_L = np.stack(projection_var_L)

    n_samples = elta_E.shape[1]
    mean_stats = np.full((n_samples, 2), np.nan)
    var_stats = np.full((n_samples, 2), np.nan)
    for t in range(n_samples):
        s, p = paired_test(elta_E[:, t], elta_L[:, t])
        mean_stats[t] = float(s), float(p)
        s, p = paired_test(eltv_E[:, t], eltv_L[:, t])
        var_stats[t] = float(s), float(p)

    projection_mean_stats, projection_var_stats = [], []
    for c in range(n_components):
        s, p = paired_test(projection_mean_E[:, c], projection_mean_L[:, c])
        projection_mean_stats.append((float(s), float(p)))
        s, p = paired_test(projection_var_E[:, c], projection_var_L[:, c])
        projection_var_stats.append((float(s), float(p)))

    return {
        'subjs': subjs,
        'elta_E': elta_E, 'elta_L': elta_L,
        'eltv_E': eltv_E, 'eltv_L': eltv_L,
        'mean_stats': mean_stats, 'var_stats': var_stats,
        'pc_components': pc_components,
        'pc_explained_var': pc_explained,
        'projection_mean_E': projection_mean_E, 'projection_mean_L': projection_mean_L,
        'projection_var_E': projection_var_E, 'projection_var_L': projection_var_L,
        'projection_mean_stats': projection_mean_stats, 'projection_var_stats': projection_var_stats,
    }


def quantify_hazard_rates(hazard: dict,
                          config=ANALYSIS_OPTIONS,
                          sig_test: str = 'wilcoxon',
                          min_n: int = 100):
    """
    Simple bin-wise test to see time-range of signficant differences.

    args:
        hazard: dict; from extract_hazard_rates
        sig_test: 'ttest' or 'wilcoxon'
        min_n: float; minimum number of trials per bin to consider valid. Only time
               bins with at least this many trials in BOTH blocks are considered.
    """

    animals = list(hazard.keys())
    bin_centres = hazard[animals[0]]['binCentres']
    n_bins = len(bin_centres)

    early = np.full((len(animals), n_bins), np.nan)
    late  = np.full((len(animals), n_bins), np.nan)
    diffs = np.full((len(animals), n_bins), np.nan)

    for a, animal in enumerate(animals):
        haz = hazard[animal]
        eValid = haz['early_n'] >= min_n
        lValid = haz['late_n'] >= min_n
        both   = eValid & lValid

        early[a, eValid] = haz['earlyBlock'][eValid]
        late[a, lValid]  = haz['lateBlock'][lValid]
        diffs[a, both]   = (haz['earlyBlock'] - haz['lateBlock'])[both]

    stat = np.full(n_bins, np.nan)
    ps   = np.full(n_bins, np.nan)
    if sig_test == 'ttest':
        from scipy.stats import ttest_1samp
        stat_fn = lambda d: ttest_1samp(d, 0, nan_policy='omit')
    elif sig_test == 'wilcoxon':
        from scipy.stats import wilcoxon as stat_fn
    else:
        raise ValueError(f'unknown sig_test: {sig_test}')

    for t in range(n_bins):
        col = diffs[:, t]
        if np.sum(~np.isnan(col)) < 2:
            continue
        stat[t], ps[t] = stat_fn(col)

    return {
        'binCentres': bin_centres,
        'animals': animals,
        'early': early,
        'late': late,
        'diffs': diffs,
        'stat': stat,
        'ps': ps,
        'sig_test': sig_test,
        'min_n': min_n,
    }


def quantify_pulse_lick_probability(pulse_lick: dict,
                                    config=ANALYSIS_OPTIONS,
                                    sig_test: str = 'wilcoxon',
                                    min_n: int = 500):
    """
    Per-animal OLS fit of P(lick) ~ b0 + b1 * tf_dev to baseline-pulse-aligned
    lick probability, per block x time window. Then paired test between blocks
    on bias and slope across animals, per time window.

    args:
        pulse_lick: dict from calculate_pulse_lick_prob
        sig_test: 'ttest' or 'wilcoxon'
        min_n: minimum pulses per bin (per animal) for that bin to enter the fit
    """
    if sig_test == 'ttest':
        from scipy.stats import ttest_rel
        paired_test = ttest_rel
    elif sig_test == 'wilcoxon':
        from scipy.stats import wilcoxon as paired_test
    else:
        raise ValueError(f'unknown sig_test: {sig_test}')

    subjs = list(pulse_lick.keys())
    first = pulse_lick[subjs[0]]
    bin_centres = first['binCentres']
    time_starts = first['time_starts']
    time_win = first['time_win']

    def fit_subj(p, n):
        valid = (n >= min_n) & ~np.isnan(p)
        if valid.sum() < 2:
            return np.nan, np.nan
        x = bin_centres[valid]
        y = p[valid]
        b1, b0 = np.polyfit(x, y, 1)
        return b0, b1

    results = {}
    for t_start in time_starts:
        t_end = t_start + time_win
        label = f'{t_start:.0f}-{t_end:.0f}s'

        block_fits = {}
        for block in ('early', 'late'):
            key = f'{block}Block_{label}'
            b0s = np.full(len(subjs), np.nan)
            b1s = np.full(len(subjs), np.nan)
            for i, s in enumerate(subjs):
                if key not in pulse_lick[s]:
                    continue
                p = pulse_lick[s][key]['lickProb']
                n = pulse_lick[s][key]['n']
                b0s[i], b1s[i] = fit_subj(p, n)
            block_fits[block] = {'bias': b0s, 'slope': b1s}

        stats = {}
        for param in ('bias', 'slope'):
            e = block_fits['early'][param]
            l = block_fits['late'][param]
            paired = ~np.isnan(e) & ~np.isnan(l)
            if paired.sum() < 2:
                stats[param] = (np.nan, np.nan)
                continue
            stat, p = paired_test(e[paired], l[paired])
            stats[param] = (float(stat), float(p))

        results[label] = {
            'early': block_fits['early'],
            'late': block_fits['late'],
            'stats': stats,
        }

    return {
        'subjs': subjs,
        'time_starts': time_starts,
        'time_win': time_win,
        'sig_test': sig_test,
        'min_n': min_n,
        'by_window': results,
    }



def quantify_integration_time(interaction: dict,
                              config=ANALYSIS_OPTIONS,
                              sig_test: str = 'ttest',
                              min_n: int = 20,
                              alpha: float = 0.05):
    """
    quantify behavioural integration timescale from the two-pulse interaction
    index J(delay), per block. J(d) is the relative facilitation of P(lick) by
    a second TF pulse at delay d, over the prediction from independent pulse
    effects (paper Methods, denoted I): J=0 means pulses act independently,
    J>0 means evidence from the first pulse is still influencing the decision.

    per delay bin: one-sample test across animals on J vs 0. integration time
    per block = largest delay where J is sig > 0. per animal per block: fit
    J(d) = A * exp(-d/tau) for a decay timescale; paired test on tau between
    blocks.

    args:
        interaction: dict from compute_interaction_index, keyed by subject
        sig_test: 'ttest' or 'wilcoxon'
        min_n: min n_pairs per delay bin (per animal) for that bin to count
        alpha: significance threshold for "integration time" cutoff
    """
    from scipy.optimize import curve_fit
    from scipy.stats import ttest_1samp, ttest_rel, wilcoxon

    if sig_test == 'ttest':
        one_sample = lambda d: ttest_1samp(d, 0, nan_policy='omit')
        paired_test = ttest_rel
    elif sig_test == 'wilcoxon':
        one_sample = wilcoxon
        paired_test = wilcoxon
    else:
        raise ValueError(f'unknown sig_test: {sig_test}')

    subjs = [s for s in interaction
             if isinstance(interaction[s], dict)
             and ('early' in interaction[s] or 'late' in interaction[s])]
    delay_centres = interaction[subjs[0]]['delay_centres']
    n_delays = len(delay_centres)

    def exp_decay(d, A, tau):
        return A * np.exp(-d / tau)

    block_results = {}
    for block in ('early', 'late'):
        # stack J across animals, masking delay bins with too few pulse pairs
        j_per_subj = np.full((len(subjs), n_delays), np.nan)
        for i, s in enumerate(subjs):
            if block not in interaction[s]:
                continue
            data = interaction[s][block]
            j_subj = data['J'].copy()
            j_subj[data['n_pairs'] < min_n] = np.nan
            j_per_subj[i] = j_subj

        stat = np.full(n_delays, np.nan)
        pvals = np.full(n_delays, np.nan)
        for d in range(n_delays):
            j_across_subjs = j_per_subj[:, d]
            j_across_subjs = j_across_subjs[~np.isnan(j_across_subjs)]
            if len(j_across_subjs) < 2:
                continue
            stat[d], pvals[d] = one_sample(j_across_subjs)

        # integration time = largest delay where J is sig > 0 across animals
        sig_pos = (pvals < alpha) & (np.nanmean(j_per_subj, axis=0) > 0)
        integration_time = (float(delay_centres[np.where(sig_pos)[0].max()])
                            if sig_pos.any() else np.nan)

        # per-animal exponential decay fit on J(delay)
        amps = np.full(len(subjs), np.nan)
        taus = np.full(len(subjs), np.nan)
        for i in range(len(subjs)):
            j_subj = j_per_subj[i]
            valid = ~np.isnan(j_subj)
            if valid.sum() < 3:
                continue
            amp_init = float(np.clip(j_subj[valid][0], 0.05, 9.9))
            try:
                (amps[i], taus[i]), _ = curve_fit(
                    exp_decay, delay_centres[valid], j_subj[valid],
                    p0=[amp_init, 0.2],
                    bounds=([0, 1e-3], [10, 5]),
                    maxfev=2000)
            except Exception as e:
                print(f'tau fit failed for {subjs[i]} ({block}): '
                      f'{type(e).__name__}: {e}')

        n_valid = np.sum(~np.isnan(j_per_subj), axis=0).astype(float)
        block_results[block] = {
            'J': j_per_subj,
            'J_mean': np.nanmean(j_per_subj, axis=0),
            'J_sem': np.nanstd(j_per_subj, axis=0) / np.sqrt(np.maximum(n_valid, 1)),
            'stat': stat,
            'pval': pvals,
            'integration_time': integration_time,
            'amp': amps,
            'tau': taus,
        }

    # paired E vs L on tau
    eT = block_results['early']['tau']
    lT = block_results['late']['tau']
    paired = ~np.isnan(eT) & ~np.isnan(lT)
    if paired.sum() >= 2:
        s, p = paired_test(eT[paired], lT[paired])
        tau_stats = (float(s), float(p))
    else:
        tau_stats = (np.nan, np.nan)

    return {
        'subjs': subjs,
        'delay_centres': delay_centres,
        'min_n': min_n,
        'sig_test': sig_test,
        'alpha': alpha,
        'early': block_results['early'],
        'late': block_results['late'],
        'tau_stats': tau_stats,
    }


def run_all_quantifications(config=ANALYSIS_OPTIONS, overwrite=False):

    stats_path = os.path.join(BEHAVIOUR_DATA_DIR, 'stats.pkl')
    if not overwrite and os.path.exists(stats_path):
        return load_behavioural('stats')

    psycho, chrono, n_hits, n_trials = load_behavioural('psychometric')
    elts = load_behavioural('elts')
    hazard = load_behavioural('hazard_rates')
    pulse_lick = load_behavioural('pulse_lick_prob')
    two_pulse = load_behavioural('two_pulse_interaction')

    stats = {
        'psychometric': quantify_change_detection(psycho, chrono, n_hits, n_trials, config),
        'lts': quantify_lick_triggered_stim(elts, config),
        'hazard_rates': quantify_hazard_rates(hazard, config),
        'pulse_lick_prob': quantify_pulse_lick_probability(pulse_lick, config),
        'integration_time': quantify_integration_time(two_pulse, config),
    }
    save_behavioural(stats, 'stats')
    return stats


