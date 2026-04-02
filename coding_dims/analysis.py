"""coding dimension analysis: statistics, alignment, projections"""

import numpy as np
import pickle
from pathlib import Path

from config import PATHS, CODING_DIM_OPS
from coding_dims.extract import _file_suffix, cosine_similarity
from data.session import Session
from utils.time import window_label

#%% between-block rotation analysis

def load_dimension_results(dim_type, npx_dir=PATHS['npx_dir_local'],
                           area=None, unit_filter=None):
    """load extracted tf or motor dimension results from pickle"""
    save_dir = Path(npx_dir) / 'coding_dims'
    suffix = _file_suffix(area, unit_filter)
    with open(save_dir / f'{dim_type}_dimensions_{suffix}.pkl', 'rb') as f:
        return pickle.load(f)


def per_animal_significance(results):
    """
    compute per-animal between-block cosine similarity and empirical p-value.
    returns dict keyed by window label, each with parallel arrays across animals
    """
    windows = set()
    for animal_res in results.values():
        windows.update(animal_res['between_block_cosine'].keys())

    out = {}
    for wl in sorted(windows):
        animals, cosines, p_values, nulls = [], [], [], []
        for animal, animal_res in sorted(results.items()):
            bc = animal_res['between_block_cosine'].get(wl)
            if bc is None:
                continue
            real = bc['real']
            null = bc['null']
            valid_null = null[~np.isnan(null)]
            p = np.mean(valid_null <= real) if len(valid_null) > 0 else np.nan

            animals.append(animal)
            cosines.append(real)
            p_values.append(p)
            nulls.append(null)

        out[wl] = dict(
            animals=animals,
            cosines=np.array(cosines),
            p_values=np.array(p_values),
            nulls=nulls,
        )
    return out


def pooled_null_test(results, n_perm=10000):
    """
    population-level significance test for between-block consistency. for each
    permutation: sample one null value per animal and average.
    builds a null distribution of population-mean cosine similarities.
    """
    windows = set()
    for animal_res in results.values():
        windows.update(animal_res['between_block_cosine'].keys())

    rng = np.random.default_rng(0)
    out = {}
    for wl in sorted(windows):
        # collect real and null per animal
        real_vals, null_arrays = [], []
        for animal, animal_res in sorted(results.items()):
            bc = animal_res['between_block_cosine'].get(wl)
            if bc is None:
                continue
            real_vals.append(bc['real'])
            null_arrays.append(bc['null'])

        if not real_vals:
            continue

        observed_mean = np.nanmean(real_vals)

        # build null distribution of population means
        null_means = np.full(n_perm, np.nan)
        for p in range(n_perm):
            sampled = []
            for null in null_arrays:
                valid = null[~np.isnan(null)]
                if len(valid) > 0:
                    sampled.append(rng.choice(valid))
            if sampled:
                null_means[p] = np.mean(sampled)

        valid_null_means = null_means[~np.isnan(null_means)]
        p_value = np.mean(valid_null_means <= observed_mean) if len(valid_null_means) > 0 else np.nan

        out[wl] = dict(
            observed_mean=observed_mean,
            null_means=null_means,
            p_value=p_value,
            n_animals=len(real_vals),
        )
    return out


def pooled_pseudopop_cosine_test(results, dim_type, n_perm=500):
    """
    pooled pseudo-population between-block cosine test. concatenates per-session
    intermediate data across all animals, computes directions on the full
    population, and tests between-block cosine with within-session shuffle null.
    dim_type: 'tf' or 'motor'
    """
    # collect per-session intermediate data across all animals
    if dim_type == 'tf':
        all_sess_tavg = {}  # {wl: {block: {polarity: [(nEv, nN_sess), ...]}}}
        all_sess_n = {}     # {block: {polarity: [int, ...]}}

        for animal, res in sorted(results.items()):
            tavg = res.get('sess_tavg')
            sn = res.get('sess_n')
            if tavg is None or sn is None:
                continue
            for wl in tavg:
                if wl not in all_sess_tavg:
                    all_sess_tavg[wl] = {b: {p: [] for p in ['fast', 'slow']}
                                          for b in ['early', 'late']}
                for block in ['early', 'late']:
                    for polarity in ['fast', 'slow']:
                        all_sess_tavg[wl][block][polarity].extend(
                            tavg[wl][block][polarity])
            # extend sess_n once per animal, outside the window loop
            if not all_sess_n:
                all_sess_n = {b: {p: [] for p in ['fast', 'slow']}
                              for b in ['early', 'late']}
            for block in ['early', 'late']:
                for polarity in ['fast', 'slow']:
                    all_sess_n[block][polarity].extend(
                        sn[block][polarity])

        rng = np.random.default_rng(0)
        out = {}
        for wl in sorted(all_sess_tavg.keys()):
            # compute real directions on pooled population
            early_fast = [np.nanmean(s, axis=0) for s in all_sess_tavg[wl]['early']['fast']]
            early_slow = [np.nanmean(s, axis=0) for s in all_sess_tavg[wl]['early']['slow']]
            late_fast = [np.nanmean(s, axis=0) for s in all_sess_tavg[wl]['late']['fast']]
            late_slow = [np.nanmean(s, axis=0) for s in all_sess_tavg[wl]['late']['slow']]

            if not early_fast or not late_fast:
                continue

            w_early = np.concatenate(early_fast) - np.concatenate(early_slow)
            w_late = np.concatenate(late_fast) - np.concatenate(late_slow)
            ne, nl = np.linalg.norm(w_early), np.linalg.norm(w_late)
            if ne == 0 or nl == 0:
                continue
            real_cos = cosine_similarity(w_early / ne, w_late / nl)

            # null: shuffle block labels within session
            n_sess = len(all_sess_tavg[wl]['early']['fast'])
            sess_pooled_fast = []
            sess_pooled_slow = []
            n_early_fast = []
            n_early_slow = []
            for s in range(n_sess):
                sess_pooled_fast.append(np.concatenate(
                    [all_sess_tavg[wl]['early']['fast'][s],
                     all_sess_tavg[wl]['late']['fast'][s]], axis=0))
                sess_pooled_slow.append(np.concatenate(
                    [all_sess_tavg[wl]['early']['slow'][s],
                     all_sess_tavg[wl]['late']['slow'][s]], axis=0))
                n_early_fast.append(all_sess_n['early']['fast'][s])
                n_early_slow.append(all_sess_n['early']['slow'][s])

            null_cos = np.full(n_perm, np.nan)
            for p in range(n_perm):
                fa, fb, sa, sb = [], [], [], []
                for pf, ps, nef, nes in zip(
                        sess_pooled_fast, sess_pooled_slow,
                        n_early_fast, n_early_slow):
                    idx_f = rng.permutation(pf.shape[0])
                    f_shuf = pf[idx_f]
                    fa.append(np.nanmean(f_shuf[:nef], axis=0))
                    fb.append(np.nanmean(f_shuf[nef:], axis=0))
                    idx_s = rng.permutation(ps.shape[0])
                    s_shuf = ps[idx_s]
                    sa.append(np.nanmean(s_shuf[:nes], axis=0))
                    sb.append(np.nanmean(s_shuf[nes:], axis=0))

                wa = np.concatenate(fa) - np.concatenate(sa)
                wb = np.concatenate(fb) - np.concatenate(sb)
                na, nb = np.linalg.norm(wa), np.linalg.norm(wb)
                if na > 0 and nb > 0:
                    null_cos[p] = cosine_similarity(wa / na, wb / nb)

            valid = null_cos[~np.isnan(null_cos)]
            p_value = np.mean(valid <= real_cos) if len(valid) > 0 else np.nan

            out[wl] = dict(
                real_cosine=real_cos,
                null_cosines=null_cos,
                p_value=p_value,
                n_sessions=n_sess,
            )
        return out

    elif dim_type == 'motor':
        all_sess_win = {}   # {wl: {block: [(nEv, nN_sess), ...]}}
        all_sess_bl = {}    # {block: [(nEv, nN_sess), ...]}
        all_sess_n = {}     # {block: [int, ...]}

        for animal, res in sorted(results.items()):
            tw = res.get('sess_tavg_win')
            tb = res.get('sess_tavg_bl')
            sn = res.get('sess_n')
            if tw is None or tb is None or sn is None:
                continue
            for wl in tw:
                if wl not in all_sess_win:
                    all_sess_win[wl] = {b: [] for b in ['early', 'late']}
                for block in ['early', 'late']:
                    all_sess_win[wl][block].extend(tw[wl][block])
            for block in ['early', 'late']:
                all_sess_bl.setdefault(block, []).extend(tb[block])
                all_sess_n.setdefault(block, []).extend(sn[block])

        rng = np.random.default_rng(0)
        out = {}
        for wl in sorted(all_sess_win.keys()):
            # real directions
            early_win = [np.nanmean(s, axis=0) for s in all_sess_win[wl]['early']]
            early_bl = [np.nanmean(s, axis=0) for s in all_sess_bl['early']]
            late_win = [np.nanmean(s, axis=0) for s in all_sess_win[wl]['late']]
            late_bl = [np.nanmean(s, axis=0) for s in all_sess_bl['late']]

            if not early_win or not late_win:
                continue

            w_early = np.concatenate(early_win) - np.concatenate(early_bl)
            w_late = np.concatenate(late_win) - np.concatenate(late_bl)
            ne, nl = np.linalg.norm(w_early), np.linalg.norm(w_late)
            if ne == 0 or nl == 0:
                continue
            real_cos = cosine_similarity(w_early / ne, w_late / nl)

            # null: shuffle block labels within session
            n_sess = len(all_sess_win[wl]['early'])
            sess_pooled_win = []
            sess_pooled_bl = []
            n_early = []
            for s in range(n_sess):
                sess_pooled_win.append(np.concatenate(
                    [all_sess_win[wl]['early'][s],
                     all_sess_win[wl]['late'][s]], axis=0))
                sess_pooled_bl.append(np.concatenate(
                    [all_sess_bl['early'][s],
                     all_sess_bl['late'][s]], axis=0))
                n_early.append(all_sess_n['early'][s])

            null_cos = np.full(n_perm, np.nan)
            for p in range(n_perm):
                wa_w, wb_w, wa_b, wb_b = [], [], [], []
                for pw, pb, n_e in zip(sess_pooled_win, sess_pooled_bl, n_early):
                    idx = rng.permutation(pw.shape[0])
                    w_shuf = pw[idx]
                    b_shuf = pb[idx]
                    wa_w.append(np.nanmean(w_shuf[:n_e], axis=0))
                    wb_w.append(np.nanmean(w_shuf[n_e:], axis=0))
                    wa_b.append(np.nanmean(b_shuf[:n_e], axis=0))
                    wb_b.append(np.nanmean(b_shuf[n_e:], axis=0))

                wa = np.concatenate(wa_w) - np.concatenate(wa_b)
                wb = np.concatenate(wb_w) - np.concatenate(wb_b)
                na, nb = np.linalg.norm(wa), np.linalg.norm(wb)
                if na > 0 and nb > 0:
                    null_cos[p] = cosine_similarity(wa / na, wb / nb)

            valid = null_cos[~np.isnan(null_cos)]
            p_value = np.mean(valid <= real_cos) if len(valid) > 0 else np.nan

            out[wl] = dict(
                real_cosine=real_cos,
                null_cosines=null_cos,
                p_value=p_value,
                n_sessions=n_sess,
            )
        return out


def analyse_coding_dimensions(dim_type: str,
                              npx_dir: str = PATHS['npx_dir_local'],
                              bm_ops: dict = CODING_DIM_OPS,
                              area: str | None = None,
                              unit_filter: list[str] | None = None,
                              save_dir: str | None = None) -> dict:
    """load results, run stats, plot"""
    suffix = _file_suffix(area, unit_filter)
    results = load_dimension_results(dim_type, npx_dir, area, unit_filter)

    per_animal = per_animal_significance(results)
    pooled = pooled_null_test(results, n_perm=bm_ops['n_perm_across'])
    pooled_pseudopop = pooled_pseudopop_cosine_test(
        results, dim_type, n_perm=bm_ops['n_perm_pooled'])

    for wl in sorted(per_animal.keys()):
        pa = per_animal[wl]
        po = pooled.get(wl, {})
        n_sig = np.sum(pa['p_values'] < 0.05)
        pp = pooled_pseudopop.get(wl, {})
        print(f'{dim_type} [{suffix}] {wl}: '
              f'{len(pa["animals"])} animals, '
              f'{n_sig}/{len(pa["animals"])} sig at p<0.05, '
              f'mean cosine={np.nanmean(pa["cosines"]):.3f}, '
              f'across-animals p={po.get("p_value", np.nan):.4f}, '
              f'pooled-pseudopop p={pp.get("p_value", np.nan):.4f}')

    from coding_dims.plotting import plot_tf_dimensions, plot_motor_dimensions
    plot_fn = plot_tf_dimensions if dim_type == 'tf' else plot_motor_dimensions
    plot_kwargs = dict(npx_dir=npx_dir, area=area, unit_filter=unit_filter)
    if save_dir is not None:
        plot_kwargs['save_dir'] = save_dir
    plot_fn(**plot_kwargs)

    return dict(per_animal=per_animal, pooled=pooled,
                pooled_pseudopop=pooled_pseudopop)


# separate fn - testing significance of projected data rather than cosine sim for block dims
def analyse_block_dimensions(npx_dir: str = PATHS['npx_dir_local'],
                             bm_ops: dict = CODING_DIM_OPS,
                             area: str | None = None,
                             unit_filter: list[str] | None = None) -> dict:
    """
    block coding dimension significance tests: per-animal AUC ROC, across-animals mean AUC, and pooled
    pseudo-population AUC
    """
    from utils.shuffle import circular_shift_labels
    from utils.stats import roc_auc

    suffix = _file_suffix(area, unit_filter)
    results = load_dimension_results('block', npx_dir, area, unit_filter)

    windows = bm_ops['block_coding_windows']

    # per-animal results (already computed during extraction)
    per_animal = {}
    for wl in [window_label(w) for w in windows]:
        animals, aucs, p_values, null_aucs_list = [], [], [], []
        for animal, res in sorted(results.items()):
            real_auc = res.get('real_aucs', {}).get(wl)
            if real_auc is None:
                continue
            animals.append(animal)
            aucs.append(real_auc)
            p_values.append(res.get('p_values', {}).get(wl, np.nan))
            null_aucs_list.append(res.get('null_aucs', {}).get(wl, np.array([])))

        if animals:
            per_animal[wl] = dict(
                animals=animals,
                aucs=np.array(aucs),
                p_values=np.array(p_values),
                null_aucs=null_aucs_list,
            )

    # across-animals test: mean AUC, null from sampling per-animal nulls
    n_perm_across = bm_ops['n_perm_across']
    rng = np.random.default_rng(0)
    across_animals = {}
    for wl, pa in per_animal.items():
        observed_mean = np.nanmean(pa['aucs'])
        null_means = np.full(n_perm_across, np.nan)
        for p in range(n_perm_across):
            sampled = []
            for null in pa['null_aucs']:
                valid = null[~np.isnan(null)]
                if len(valid) > 0:
                    sampled.append(rng.choice(valid))
            if sampled:
                null_means[p] = np.mean(sampled)

        valid_null = null_means[~np.isnan(null_means)]
        p_value = np.mean(valid_null >= observed_mean) if len(valid_null) > 0 else np.nan

        across_animals[wl] = dict(
            observed_mean=observed_mean,
            null_means=null_means,
            p_value=p_value,
            n_animals=len(pa['animals']),
        )

    # pooled pseudo-population test: concatenate sessions, held-out AUC
    from coding_dims.extract import (_compute_block_direction, _project_test_auc,
                                     _get_fit_labels)

    n_perm_pooled = bm_ops['n_perm_pooled']

    # collect all per-session intermediate data across animals
    all_trial_lists = []
    all_labels = []
    all_fit_idx = []
    all_test_idx = []
    all_n_neurons = []

    for animal, res in sorted(results.items()):
        sess_int = res.get('sess_intermediate')
        if sess_int is None:
            continue
        for si in sess_int:
            all_trial_lists.append(si['trial_list'])
            all_labels.append(si['all_block_labels'])
            all_fit_idx.append(si['fit_idx'])
            all_test_idx.append(si['test_idx'])
            all_n_neurons.append(si['n_neurons'])

    neuron_offsets = np.cumsum([0] + all_n_neurons)
    rng_pooled = np.random.default_rng(0)
    pooled = {}

    real_fit_labels = _get_fit_labels(all_trial_lists, all_fit_idx)

    for win in windows:
        wl = window_label(win)

        # real AUC
        w, _ = _compute_block_direction(all_trial_lists, all_fit_idx,
                                        real_fit_labels, wl)
        if w is None:
            continue
        real_auc = _project_test_auc(all_trial_lists, all_test_idx,
                                     w, neuron_offsets, wl)

        # circular-shift null
        null_aucs_arr = np.full(n_perm_pooled, np.nan)
        for p in range(n_perm_pooled):
            shifted = [circular_shift_labels(al, rng_pooled) for al in all_labels]
            shifted_fit_labels = _get_fit_labels(all_trial_lists, all_fit_idx,
                                                 sess_all_labels=shifted)
            w_null, _ = _compute_block_direction(all_trial_lists, all_fit_idx,
                                                 shifted_fit_labels, wl)
            if w_null is not None:
                null_aucs_arr[p] = _project_test_auc(all_trial_lists, all_test_idx,
                                                     w_null, neuron_offsets, wl)

        valid = null_aucs_arr[~np.isnan(null_aucs_arr)]
        p_value = np.mean(valid >= real_auc) if len(valid) > 0 else np.nan

        pooled[wl] = dict(
            real_auc=real_auc,
            null_aucs=null_aucs_arr,
            p_value=p_value,
            n_sessions=len(all_trial_lists),
        )

    for wl in sorted(per_animal.keys()):
        pa = per_animal[wl]
        aa = across_animals.get(wl, {})
        po = pooled.get(wl, {})
        n_sig = np.sum(pa['p_values'] < 0.05)
        print(f'block [{suffix}] {wl}: '
              f'{len(pa["animals"])} animals, '
              f'{n_sig}/{len(pa["animals"])} sig at p<0.05, '
              f'mean AUC={np.nanmean(pa["aucs"]):.3f}, '
              f'across-animals p={aa.get("p_value", np.nan):.4f}, '
              f'pooled p={po.get("p_value", np.nan):.4f}')

    return dict(per_animal=per_animal, across_animals=across_animals, pooled=pooled)


#%% tf-motor alignment

def _load_mean_responses(animal, included_sessions, npx_dir, area, unit_filter, bm_ops):
    """load neuron-masked and session-concatenated mean responses to tf/licks"""
    from coding_dims.extract import _get_neuron_mask, _get_window_bins
    from data.load_responses import load_psth_mean
    from utils.smoothing import causal_boxcar

    window_bins = _get_window_bins(bm_ops)

    tf_conditions = [
        'earlyBlock_early_pos', 'earlyBlock_early_neg',
        'lateBlock_early_pos', 'lateBlock_early_neg',
        'lateBlock_late_pos', 'lateBlock_late_neg',
    ]
    lick_conditions = [
        'earlyBlock_early_fa', 'earlyBlock_early_hit',
        'lateBlock_early_fa', 'lateBlock_early_hit',
        'lateBlock_late_fa', 'lateBlock_late_hit',
    ]
    bl_conditions = ['early', 'late']

    means = {}
    t_axes = {}
    unit_ids = []

    for sess_name in included_sessions:
        sess_dir = Path(npx_dir) / animal / sess_name
        psth_path = str(sess_dir / 'psths.h5')
        neuron_mask = _get_neuron_mask(sess_dir, area, unit_filter)

        session = Session.load(str(sess_dir / 'session.pkl'))
        cids = session.unit_info['cluster_id'].values[neuron_mask]
        unit_ids.extend([(sess_name, int(c)) for c in cids])

        for event_type, conditions in [('tf', tf_conditions),
                                        ('lick', lick_conditions),
                                        ('blOn', bl_conditions)]:
            for cond in conditions:
                mean, _, t_ax = load_psth_mean(psth_path, event_type, cond,
                                               baseline_subtract=False)
                if mean is None:
                    continue
                key = f'{event_type}/{cond}'
                t_axes[event_type] = t_ax
                masked = mean[neuron_mask]
                smoothed = causal_boxcar(masked, window_bins, axis=-1)
                means.setdefault(key, []).append(smoothed)

    # concatenate across sessions (n_neurons dimension)
    concat_means = {}
    for key, session_means in means.items():
        concat_means[key] = np.concatenate(session_means, axis=0)

    return concat_means, t_axes, unit_ids


def calculate_tf_motor_alignment(npx_dir=PATHS['npx_dir_local'],
                                 bm_ops=CODING_DIM_OPS,
                                 area: None | str = None,
                                 unit_filter: None | list[str] = None):
    """
    compare TF and motor coding directions: cosine similarity between all pairs per
    block. also computes TF response projections onto motor dimensions for plotting.
    """
    save_dir = Path(npx_dir) / 'coding_dims'
    suffix = _file_suffix(area, unit_filter)

    tf_results = load_dimension_results('tf', npx_dir, area, unit_filter)
    motor_results = load_dimension_results('motor', npx_dir, area, unit_filter)

    animals = set(tf_results.keys()) & set(motor_results.keys())
    all_results = {}

    for animal in sorted(animals):
        tf_r = tf_results[animal]
        motor_r = motor_results[animal]

        # match neurons by unit_ids, subset, renormalise
        tf_ids = tf_r.get('unit_ids', [])
        motor_ids = motor_r.get('unit_ids', [])
        if not tf_ids or not motor_ids:
            print(f'  {animal}: no unit_ids saved! re-run extraction')
            continue

        tf_id_set = set(tf_ids)
        motor_id_set = set(motor_ids)
        shared = tf_id_set & motor_id_set

        if len(shared) < bm_ops.get('min_neurons', 5):
            print(f'  {animal}: only {len(shared)} shared neurons - skipping')
            continue

        tf_idx = np.array([i for i, uid in enumerate(tf_ids) if uid in shared])
        motor_idx = np.array([i for i, uid in enumerate(motor_ids) if uid in shared])

        # reorder motor_idx to match tf neuron order
        tf_shared_order = [tf_ids[i] for i in tf_idx]
        motor_id_to_idx = {uid: i for i, uid in enumerate(motor_ids)}
        motor_idx = np.array([motor_id_to_idx[uid] for uid in tf_shared_order])

        n_tf = len(tf_ids)
        n_motor = len(motor_ids)
        n_shared = len(shared)
        if n_shared < n_tf or n_shared < n_motor:
            print(f'  {animal}: {n_shared}/{n_tf} TF, {n_shared}/{n_motor} motor neurons shared')

        # cosine similarity between all TF x motor dimension pairs (shared neurons)
        alignment = {}
        for block in ['early', 'late']:
            tf_dims = tf_r['dimensions'].get(block, {})
            motor_dims = motor_r['dimensions'].get(block, {})
            alignment[block] = {}
            for tf_wl, tf_w in tf_dims.items():
                tf_sub = tf_w[tf_idx]
                for motor_wl, motor_w in motor_dims.items():
                    motor_sub = motor_w[motor_idx]
                    key = f'tf_{tf_wl}_x_motor_{motor_wl}'
                    alignment[block][key] = cosine_similarity(tf_sub, motor_sub)

        # project TF responses onto motor dimensions (shared neurons only)
        shared_sessions = sorted(set(tf_r['included_sessions']) &
                                  set(motor_r['included_sessions']))
        mean_resps, t_axes, resp_ids = _load_mean_responses(
            animal, shared_sessions, npx_dir, area, unit_filter, bm_ops)

        # reorder resp rows to match motor_idx neuron order
        resp_id_to_idx = {uid: i for i, uid in enumerate(resp_ids)}

        # only keep neurons present in all three: tf, motor, and response data
        keep = [uid for uid in tf_shared_order if uid in resp_id_to_idx]
        resp_reorder = np.array([resp_id_to_idx[uid] for uid in keep])
        motor_reorder = np.array([motor_id_to_idx[uid] for uid in keep])

        tf_onto_motor = {}
        for block in ['early', 'late']:
            motor_dims = motor_r['dimensions'].get(block, {})
            tf_onto_motor[block] = {}
            for motor_wl, motor_w in motor_dims.items():
                motor_sub = motor_w[motor_reorder]
                tf_onto_motor[block][motor_wl] = {}
                for cond_key, resp in mean_resps.items():
                    if cond_key.startswith('tf/'):
                        resp_sub = resp[resp_reorder]
                        tf_onto_motor[block][motor_wl][cond_key] = motor_sub @ resp_sub

        all_results[animal] = dict(
            alignment=alignment,
            tf_onto_motor=tf_onto_motor,
            tf_t_ax=t_axes.get('tf'),
        )

    out_path = save_dir / f'alignment_{suffix}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved alignment to {out_path}')

    return all_results


#%% cross-dimension analysis

def cross_dimension_cosines(npx_dir=PATHS['npx_dir_local'],
                            bm_ops=CODING_DIM_OPS,
                            area: str | None = None,
                            unit_filter: list[str] | None = None,
                            n_perm: int = 500):
    """
    pairwise cosine similarities between block, tf, and motor coding dimensions, with permutation p-values. neurons
    are matched by unit_ids intersection
    """
    save_dir = Path(npx_dir) / 'coding_dims'
    suffix = _file_suffix(area, unit_filter)

    block_results = load_dimension_results('block', npx_dir, area, unit_filter)
    tf_results = load_dimension_results('tf', npx_dir, area, unit_filter)
    motor_results = load_dimension_results('motor', npx_dir, area, unit_filter)

    animals = sorted(set(block_results.keys()) &
                     set(tf_results.keys()) &
                     set(motor_results.keys()))

    rng = np.random.default_rng(0)
    all_results = {}

    for animal in animals:
        block_r = block_results[animal]
        tf_r = tf_results[animal]
        motor_r = motor_results[animal]

        block_ids = block_r.get('unit_ids', [])
        tf_ids = tf_r.get('unit_ids', [])
        motor_ids = motor_r.get('unit_ids', [])
        if not block_ids or not tf_ids or not motor_ids:
            print(f'  {animal}: missing unit_ids - skipping')
            continue

        shared = set(block_ids) & set(tf_ids) & set(motor_ids)
        if len(shared) < bm_ops.get('min_neurons', 5):
            print(f'  {animal}: only {len(shared)} shared neurons - skipping')
            continue

        # build index arrays to reorder each to a common shared neuron ordering
        shared_order = sorted(shared)
        block_id_to_idx = {uid: i for i, uid in enumerate(block_ids)}
        tf_id_to_idx = {uid: i for i, uid in enumerate(tf_ids)}
        motor_id_to_idx = {uid: i for i, uid in enumerate(motor_ids)}
        block_idx = np.array([block_id_to_idx[uid] for uid in shared_order])
        tf_idx = np.array([tf_id_to_idx[uid] for uid in shared_order])
        motor_idx = np.array([motor_id_to_idx[uid] for uid in shared_order])

        # collect all named dimensions with their vectors (subset to shared neurons)
        dims = {}
        for wl, w in block_r['dimensions'].items():
            dims[f'block_{wl}'] = w[block_idx]
        for block in ['early', 'late']:
            for wl, w in tf_r['dimensions'].get(block, {}).items():
                dims[f'tf_{block}_{wl}'] = w[tf_idx]
            for wl, w in motor_r['dimensions'].get(block, {}).items():
                dims[f'motor_{block}_{wl}'] = w[motor_idx]

        dim_names = sorted(dims.keys())
        n_dims = len(dim_names)

        # pairwise cosine similarity matrix
        cosine_matrix = np.full((n_dims, n_dims), np.nan)
        p_matrix = np.full((n_dims, n_dims), np.nan)

        for i in range(n_dims):
            for j in range(i, n_dims):
                vi = dims[dim_names[i]]
                vj = dims[dim_names[j]]
                real = cosine_similarity(vi, vj)
                cosine_matrix[i, j] = real
                cosine_matrix[j, i] = real

                if i == j:
                    p_matrix[i, j] = 0.0
                    continue

                # permutation test (two-tailed): shuffle one vector's entries
                null_cos = np.full(n_perm, np.nan)
                for p in range(n_perm):
                    shuffled = rng.permutation(vi)
                    null_cos[p] = cosine_similarity(shuffled, vj)

                valid_null = null_cos[~np.isnan(null_cos)]
                if len(valid_null) > 0:
                    p_val = np.mean(np.abs(valid_null) >= np.abs(real))
                else:
                    p_val = np.nan
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val

        all_results[animal] = dict(
            cosine_matrix=cosine_matrix,
            p_matrix=p_matrix,
            dim_names=dim_names,
            n_shared=len(shared),
        )
        print(f'  {animal}: {n_dims} dimensions, {len(shared)} shared neurons')

    out_path = save_dir / f'cross_dimension_cosines_{suffix}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved cross-dimension cosines to {out_path}')

    return all_results


def cross_dimension_projections(npx_dir=PATHS['npx_dir_local'],
                                bm_ops=CODING_DIM_OPS,
                                area: str | None = None,
                                unit_filter: list[str] | None = None):
    """
    project event-aligned mean responses onto all block, tf, motor coding dimensions.
    neurons matched by unit_ids intersection across all three types
    """
    save_dir = Path(npx_dir) / 'coding_dims'
    suffix = _file_suffix(area, unit_filter)

    block_results = load_dimension_results('block', npx_dir, area, unit_filter)
    tf_results = load_dimension_results('tf', npx_dir, area, unit_filter)
    motor_results = load_dimension_results('motor', npx_dir, area, unit_filter)

    animals = sorted(set(block_results.keys()) &
                     set(tf_results.keys()) &
                     set(motor_results.keys()))

    all_results = {}

    for animal in animals:
        block_r = block_results[animal]
        tf_r = tf_results[animal]
        motor_r = motor_results[animal]

        block_ids = block_r.get('unit_ids', [])
        tf_ids = tf_r.get('unit_ids', [])
        motor_ids = motor_r.get('unit_ids', [])
        if not block_ids or not tf_ids or not motor_ids:
            print(f'  {animal}: missing unit_ids - skipping')
            continue

        shared = set(block_ids) & set(tf_ids) & set(motor_ids)
        if len(shared) < bm_ops.get('min_neurons', 5):
            print(f'  {animal}: only {len(shared)} shared neurons - skipping')
            continue

        # shared sessions across all three dimension types
        shared_sessions = sorted(
            set(block_r['included_sessions']) &
            set(tf_r['included_sessions']) &
            set(motor_r['included_sessions'])
        )
        if not shared_sessions:
            print(f'  {animal}: no shared sessions - skipping')
            continue

        # build index maps for each dimension type
        shared_order = sorted(shared)
        block_id_to_idx = {uid: i for i, uid in enumerate(block_ids)}
        tf_id_to_idx = {uid: i for i, uid in enumerate(tf_ids)}
        motor_id_to_idx = {uid: i for i, uid in enumerate(motor_ids)}
        block_idx = np.array([block_id_to_idx[uid] for uid in shared_order])
        tf_idx = np.array([tf_id_to_idx[uid] for uid in shared_order])
        motor_idx = np.array([motor_id_to_idx[uid] for uid in shared_order])

        # collect all named dimensions (subset to shared neurons)
        dims = {}
        for wl, w in block_r['dimensions'].items():
            dims[f'block_{wl}'] = w[block_idx]
        for block in ['early', 'late']:
            for wl, w in tf_r['dimensions'].get(block, {}).items():
                dims[f'tf_{block}_{wl}'] = w[tf_idx]
            for wl, w in motor_r['dimensions'].get(block, {}).items():
                dims[f'motor_{block}_{wl}'] = w[motor_idx]

        # load mean responses and match to shared neurons
        mean_resps, t_axes, resp_ids = _load_mean_responses(
            animal, shared_sessions, npx_dir, area, unit_filter, bm_ops)

        resp_id_to_idx = {uid: i for i, uid in enumerate(resp_ids)}
        keep = [uid for uid in shared_order if uid in resp_id_to_idx]

        if len(keep) < bm_ops.get('min_neurons', 5):
            print(f'  {animal}: only {len(keep)} neurons in responses - skipping')
            continue

        resp_reorder = np.array([resp_id_to_idx[uid] for uid in keep])
        # also reorder dimension vectors to match the keep order
        keep_set = set(keep)
        keep_in_shared = np.array([i for i, uid in enumerate(shared_order)
                                   if uid in keep_set])

        # project each response onto each dimension
        projections = {}
        for dim_name, w in dims.items():
            w_sub = w[keep_in_shared]
            projections[dim_name] = {}
            for resp_key, resp in mean_resps.items():
                resp_sub = resp[resp_reorder]
                projections[dim_name][resp_key] = w_sub @ resp_sub

        all_results[animal] = dict(
            projections=projections,
            t_axes=t_axes,
            dim_names=sorted(dims.keys()),
            n_shared=len(keep),
        )
        print(f'  {animal}: projected {len(mean_resps)} responses onto '
              f'{len(dims)} dimensions ({len(keep)} neurons)')

    out_path = save_dir / f'cross_dimension_projections_{suffix}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved cross-dimension projections to {out_path}')

    return all_results


#%% run all combos

def run_all_coding_dim_analyses(npx_dir=PATHS['npx_dir_local'],
                                bm_ops=CODING_DIM_OPS,
                                save_dir=None):
    """
    run extraction, analysis, and plotting for all area/unit_filter combos.
    extracts tf + motor dimensions, computes alignment, analyses + plots each.
    """
    from coding_dims.extract import extract_tf_dimensions, extract_motor_dimensions
    from coding_dims.plotting import plot_alignment

    combos = [
        dict(area=None, unit_filter=None),
        dict(area=None, unit_filter=['tf']),
        dict(area=None, unit_filter=['lick_prep']),
        dict(area=None, unit_filter=['tf', 'lick_prep']),
        dict(area='frontal_cortex', unit_filter=None),
        dict(area='basal_ganglia', unit_filter=None),
        dict(area='cerebellum', unit_filter=None),
        dict(area='early_visual', unit_filter=None),
        dict(area='higher_visual', unit_filter=None),
        dict(area='hippocampus', unit_filter=None),
    ]

    all_stats = {}
    for combo in combos:
        suffix = _file_suffix(**combo)
        print(f'\n{"=" * 60}')
        print(f'  {suffix}')
        print(f'{"=" * 60}')

        combo_save = Path(save_dir) / suffix if save_dir else None

        for dim_type, extract_fn in [('tf', extract_tf_dimensions),
                                      ('motor', extract_motor_dimensions)]:
            extract_fn(npx_dir=npx_dir, bm_ops=bm_ops, n_jobs=6, **combo)

        calculate_tf_motor_alignment(npx_dir=npx_dir, bm_ops=bm_ops, **combo)

        for dim_type in ['tf', 'motor']:
            stats = analyse_coding_dimensions(
                dim_type, npx_dir=npx_dir, bm_ops=bm_ops,
                save_dir=str(combo_save) if combo_save else None, **combo)
            all_stats[(suffix, dim_type)] = stats

        plot_alignment(npx_dir=npx_dir,
                       save_dir=str(combo_save) if combo_save else None,
                       **combo)

    return all_stats
