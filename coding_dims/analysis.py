"""coding dimension analysis: statistics, alignment, projections"""

import numpy as np
import pickle
from pathlib import Path

from config import PATHS, CODING_DIM_OPS
from utils.filing import file_suffix
from utils.stats import cosine_similarity
from data.session import Session
from utils.time import window_label

#%% between-block rotation analysis

def load_dimension_results(dim_type: str, npx_dir=PATHS['npx_dir_local'],
                           area: str | None = None,
                           unit_filter: list[str] | None = None):
    """load extracted dimension results from pickle"""
    save_dir = Path(npx_dir) / 'coding_dims'
    suffix = file_suffix(area, unit_filter)
    with open(save_dir / f'{dim_type}_dimensions_cd_{suffix}.pkl', 'rb') as f:
        return pickle.load(f)


def _list_dim_names(results):
    # infer dim_names from the first result's dimensions dict
    for animal_res in results.values():
        d = animal_res.get('dimensions')
        if isinstance(d, dict):
            return list(d.keys())
    return []


def per_animal_significance(results):
    # returns {dim_name: {win_label: {animals, cosines, p_values, nulls}}}
    dim_names = _list_dim_names(results)
    out = {v: {} for v in dim_names}
    for dim_name in dim_names:
        windows = set()
        for animal_res in results.values():
            windows.update(animal_res['between_block_cosine'].get(dim_name, {}).keys())

        for win_label in sorted(windows):
            animals, cosines, p_values, nulls = [], [], [], []
            for animal, animal_res in sorted(results.items()):
                block_cos = animal_res['between_block_cosine'].get(dim_name, {}).get(win_label)
                if block_cos is None:
                    continue
                real = block_cos['real']
                null = block_cos['null']
                valid_null = null[~np.isnan(null)]
                p = np.mean(valid_null <= real) if len(valid_null) > 0 else np.nan

                animals.append(animal)
                cosines.append(real)
                p_values.append(p)
                nulls.append(null)

            out[dim_name][win_label] = dict(
                animals=animals,
                cosines=np.array(cosines),
                p_values=np.array(p_values),
                nulls=nulls,
            )
    return out


def pooled_null_test(results, n_perm=10000):
    # returns {dim_name: {win_label: {observed_mean, null_means, p_value, n_animals}}}
    dim_names = _list_dim_names(results)
    out = {v: {} for v in dim_names}
    for dim_name in dim_names:
        windows = set()
        for animal_res in results.values():
            windows.update(animal_res['between_block_cosine'].get(dim_name, {}).keys())

        rng = np.random.default_rng(0)
        for win_label in sorted(windows):
            real_vals, null_arrays = [], []
            for animal, animal_res in sorted(results.items()):
                block_cos = animal_res['between_block_cosine'].get(dim_name, {}).get(win_label)
                if block_cos is None:
                    continue
                real_vals.append(block_cos['real'])
                null_arrays.append(block_cos['null'])

            if not real_vals:
                continue

            observed_mean = np.nanmean(real_vals)
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

            out[dim_name][win_label] = dict(
                observed_mean=observed_mean,
                null_means=null_means,
                p_value=p_value,
                n_animals=len(real_vals),
            )
    return out


def pooled_pseudopop_cosine_test(results, dim_type, n_perm=500):
    # pooled pseudo-population between-block cosine, within-session shuffle null.
    # returns {dim_name: {win_label: {real_cosine, null_cosines, p_value, n_sessions}}}
    from coding_dims.extract import TF_DIM_FNS, MOTOR_DIM_FNS

    if dim_type == 'tf':
        dim_fns = TF_DIM_FNS
        # all_a[win_label] = list of per-session (nEv, nN) arrays for 'a' class (fast)
        # all_b[win_label] = list of per-session (nEv, nN) arrays for 'b' class (slow)
        all_a, all_b = {}, {}
        all_na, all_nb = [], []  # per-session n_early (for within-session split)
        for animal, res in sorted(results.items()):
            tavg = res.get('sess_tavg')
            sn = res.get('sess_n')
            if tavg is None or sn is None:
                continue
            for win_label in tavg:
                all_a.setdefault(win_label, [])
                all_b.setdefault(win_label, [])
                for s_idx in range(len(tavg[win_label]['early']['fast'])):
                    all_a[win_label].append(np.concatenate(
                        [tavg[win_label]['early']['fast'][s_idx],
                         tavg[win_label]['late']['fast'][s_idx]], axis=0))
                    all_b[win_label].append(np.concatenate(
                        [tavg[win_label]['early']['slow'][s_idx],
                         tavg[win_label]['late']['slow'][s_idx]], axis=0))
            all_na.extend(sn['early']['fast'])
            all_nb.extend(sn['early']['slow'])

    elif dim_type == 'motor':
        dim_fns = MOTOR_DIM_FNS
        all_a, all_b = {}, {}
        all_na, all_nb = [], []
        for animal, res in sorted(results.items()):
            tw = res.get('sess_tavg_win')
            tb = res.get('sess_tavg_bl')
            sn = res.get('sess_n')
            if tw is None or tb is None or sn is None:
                continue
            for win_label in tw:
                all_a.setdefault(win_label, [])
                all_b.setdefault(win_label, [])
                for s_idx in range(len(tw[win_label]['early'])):
                    all_a[win_label].append(np.concatenate(
                        [tw[win_label]['early'][s_idx], tw[win_label]['late'][s_idx]], axis=0))
                    all_b[win_label].append(np.concatenate(
                        [tb['early'][s_idx], tb['late'][s_idx]], axis=0))
            all_na.extend(sn['early'])
            all_nb.extend(sn['early'])  # paired: same events

    out = {v: {} for v in dim_fns}
    for win_label in sorted(all_a.keys()):
        sess_a = all_a[win_label]
        sess_b = all_b[win_label]
        if not sess_a:
            continue

        # real directions per dim_name from early/late split
        real_cos = {}
        for dim_name, fn in dim_fns.items():
            w_e = np.concatenate([fn(a[:na], b[:nb])
                                  for a, b, na, nb in zip(sess_a, sess_b, all_na, all_nb)])
            w_l = np.concatenate([fn(a[na:], b[nb:])
                                  for a, b, na, nb in zip(sess_a, sess_b, all_na, all_nb)])
            real_cos[dim_name] = cosine_similarity(w_e, w_l)

        null_cos = {v: np.full(n_perm, np.nan) for v in dim_fns}
        rng = np.random.default_rng(0)
        for p in range(n_perm):
            a_a, a_b, b_a, b_b = [], [], [], []
            for sa, sb, na, nb in zip(sess_a, sess_b, all_na, all_nb):
                if dim_type == 'motor':
                    # paired: same shuffle for a and b
                    idx = rng.permutation(sa.shape[0])
                    a_shuf = sa[idx]
                    b_shuf = sb[idx]
                    a_a.append(a_shuf[:na])
                    a_b.append(a_shuf[na:])
                    b_a.append(b_shuf[:nb])
                    b_b.append(b_shuf[nb:])
                else:
                    # unpaired: independent shuffles
                    idx_a = rng.permutation(sa.shape[0])
                    a_shuf = sa[idx_a]
                    a_a.append(a_shuf[:na])
                    a_b.append(a_shuf[na:])
                    idx_b = rng.permutation(sb.shape[0])
                    b_shuf = sb[idx_b]
                    b_a.append(b_shuf[:nb])
                    b_b.append(b_shuf[nb:])

            for dim_name, fn in dim_fns.items():
                w_a = np.concatenate([fn(a, b) for a, b in zip(a_a, b_a)])
                w_b = np.concatenate([fn(a, b) for a, b in zip(a_b, b_b)])
                null_cos[dim_name][p] = cosine_similarity(w_a, w_b)

        n_sess = len(sess_a)
        for dim_name in dim_fns:
            valid = null_cos[dim_name][~np.isnan(null_cos[dim_name])]
            p_value = np.mean(valid <= real_cos[dim_name]) if len(valid) > 0 else np.nan
            out[dim_name][win_label] = dict(
                real_cosine=real_cos[dim_name],
                null_cosines=null_cos[dim_name],
                p_value=p_value,
                n_sessions=n_sess,
            )
    return out


def analyse_coding_dimensions(dim_type: str,
                              npx_dir=PATHS['npx_dir_local'],
                              cd_ops: dict = CODING_DIM_OPS,
                              area: str | None = None,
                              unit_filter: list[str] | None = None,
                              dim_name: str = 'cd',
                              save_dir=None) -> dict:
    """load results, run stats, plot"""
    suffix = file_suffix(area, unit_filter)
    results = load_dimension_results(dim_type, npx_dir, area, unit_filter)

    per_animal = per_animal_significance(results)
    pooled = pooled_null_test(results, n_perm=cd_ops['n_perm_across'])
    pooled_pseudopop = pooled_pseudopop_cosine_test(
        results, dim_type, n_perm=cd_ops['n_perm_pooled'])

    for dim_key in per_animal:
        for win_label in sorted(per_animal[dim_key].keys()):
            per_anim = per_animal[dim_key][win_label]
            pooled_stats = pooled[dim_key].get(win_label, {})
            pseudo_stats = pooled_pseudopop[dim_key].get(win_label, {})
            n_sig = np.sum(per_anim['p_values'] < 0.05)
            print(f'{dim_type} [{suffix}] [{dim_key}] {win_label}: '
                  f'{len(per_anim["animals"])} animals, '
                  f'{n_sig}/{len(per_anim["animals"])} sig at p<0.05, '
                  f'mean cosine={np.nanmean(per_anim["cosines"]):.3f}, '
                  f'across-animals p={pooled_stats.get("p_value", np.nan):.4f}, '
                  f'pooled-pseudopop p={pseudo_stats.get("p_value", np.nan):.4f}')

    from coding_dims.plotting import plot_tf_dimensions, plot_motor_dimensions
    plot_fn = plot_tf_dimensions if dim_type == 'tf' else plot_motor_dimensions
    plot_kwargs = dict(npx_dir=npx_dir, area=area, unit_filter=unit_filter, dim_name=dim_name)
    if save_dir is not None:
        plot_kwargs['save_dir'] = save_dir
    plot_fn(**plot_kwargs)

    return dict(per_animal=per_animal, pooled=pooled,
                pooled_pseudopop=pooled_pseudopop)


# separate fn - testing significance of projected data rather than cosine sim for block dims
def analyse_block_dimensions(npx_dir=PATHS['npx_dir_local'],
                             cd_ops: dict = CODING_DIM_OPS,
                             area: str | None = None,
                             unit_filter: list[str] | None = None,
                             ) -> dict:
    """
    block cd significance tests: per-animal AUC ROC, across-animals mean AUC, pooled pseudo-population AUC
    """
    from utils.shuffle import circular_shift_labels
    from coding_dims.extract import (BLOCK_DIM_FNS, _compute_block_directions,
                                     _project_test_auc, _get_fit_labels)

    suffix = file_suffix(area, unit_filter)
    results = load_dimension_results('block', npx_dir, area, unit_filter)

    windows = cd_ops['block_coding_windows']
    win_labels = [window_label(w) for w in windows]
    dim_names = list(BLOCK_DIM_FNS)

    # per-animal results (already computed during extraction, nested by dim_name)
    per_animal = {v: {} for v in dim_names}
    for dim_name in dim_names:
        for win_label in win_labels:
            animals, aucs, p_values, null_aucs_list = [], [], [], []
            for animal, res in sorted(results.items()):
                real_auc = res.get('real_aucs', {}).get(dim_name, {}).get(win_label)
                if real_auc is None:
                    continue
                animals.append(animal)
                aucs.append(real_auc)
                p_values.append(res.get('p_values', {}).get(dim_name, {}).get(win_label, np.nan))
                null_aucs_list.append(res.get('null_aucs', {}).get(dim_name, {}).get(win_label, np.array([])))

            if animals:
                per_animal[dim_name][win_label] = dict(
                    animals=animals,
                    aucs=np.array(aucs),
                    p_values=np.array(p_values),
                    null_aucs=null_aucs_list,
                )

    # across-animals test: mean AUC, null from sampling per-animal nulls
    n_perm_across = cd_ops['n_perm_across']
    across_animals = {v: {} for v in dim_names}
    for dim_name in dim_names:
        rng = np.random.default_rng(0)
        for win_label, per_anim in per_animal[dim_name].items():
            observed_mean = np.nanmean(per_anim['aucs'])
            null_means = np.full(n_perm_across, np.nan)
            for p in range(n_perm_across):
                sampled = []
                for null in per_anim['null_aucs']:
                    valid = null[~np.isnan(null)]
                    if len(valid) > 0:
                        sampled.append(rng.choice(valid))
                if sampled:
                    null_means[p] = np.mean(sampled)

            valid_null = null_means[~np.isnan(null_means)]
            p_value = np.mean(valid_null >= observed_mean) if len(valid_null) > 0 else np.nan

            across_animals[dim_name][win_label] = dict(
                observed_mean=observed_mean,
                null_means=null_means,
                p_value=p_value,
                n_animals=len(per_anim['animals']),
            )

    # pooled pseudo-population test: concatenate sessions, held-out AUC
    n_perm_pooled = cd_ops['n_perm_pooled']
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
    real_fit_labels = _get_fit_labels(all_trial_lists, all_fit_idx)

    # real AUCs per window, per dim_name
    real_aucs_pooled = {v: {} for v in dim_names}
    for win in windows:
        win_label = window_label(win)
        dirs = _compute_block_directions(all_trial_lists, all_fit_idx,
                                         real_fit_labels, win_label)
        for dim_name in dim_names:
            w, _ = dirs[dim_name]
            if w is None:
                continue
            real_aucs_pooled[dim_name][win_label] = _project_test_auc(
                all_trial_lists, all_test_idx, w, neuron_offsets, win_label)

    # circular-shift null (compute all dim_names from same shuffle)
    null_aucs_pooled = {v: {win_label: np.full(n_perm_pooled, np.nan) for win_label in win_labels}
                        for v in dim_names}
    rng_pooled = np.random.default_rng(0)
    for p in range(n_perm_pooled):
        shifted = [circular_shift_labels(al, rng_pooled) for al in all_labels]
        shifted_fit_labels = _get_fit_labels(all_trial_lists, all_fit_idx,
                                             sess_all_labels=shifted)
        for win in windows:
            win_label = window_label(win)
            dirs = _compute_block_directions(all_trial_lists, all_fit_idx,
                                             shifted_fit_labels, win_label)
            for dim_name in dim_names:
                w_null, _ = dirs[dim_name]
                if w_null is None or win_label not in real_aucs_pooled[dim_name]:
                    continue
                null_aucs_pooled[dim_name][win_label][p] = _project_test_auc(
                    all_trial_lists, all_test_idx, w_null, neuron_offsets, win_label)

    pooled = {v: {} for v in dim_names}
    for dim_name in dim_names:
        for win_label, real_auc in real_aucs_pooled[dim_name].items():
            null_arr = null_aucs_pooled[dim_name][win_label]
            valid = null_arr[~np.isnan(null_arr)]
            p_value = np.mean(valid >= real_auc) if len(valid) > 0 else np.nan
            pooled[dim_name][win_label] = dict(
                real_auc=real_auc,
                null_aucs=null_arr,
                p_value=p_value,
                n_sessions=len(all_trial_lists),
            )

    for dim_name in dim_names:
        for win_label in sorted(per_animal[dim_name].keys()):
            per_anim = per_animal[dim_name][win_label]
            across_stats = across_animals[dim_name].get(win_label, {})
            pooled_stats = pooled[dim_name].get(win_label, {})
            n_sig = np.sum(per_anim['p_values'] < 0.05)
            print(f'block [{suffix}] [{dim_name}] {win_label}: '
                  f'{len(per_anim["animals"])} animals, '
                  f'{n_sig}/{len(per_anim["animals"])} sig at p<0.05, '
                  f'mean AUC={np.nanmean(per_anim["aucs"]):.3f}, '
                  f'across-animals p={across_stats.get("p_value", np.nan):.4f}, '
                  f'pooled p={pooled_stats.get("p_value", np.nan):.4f}')

    return dict(per_animal=per_animal, across_animals=across_animals, pooled=pooled)


#%% tf-motor alignment

def _load_mean_responses(animal, included_sessions, npx_dir, area, unit_filter, cd_ops):
    """load neuron-masked and session-concatenated mean responses to tf/licks"""
    from utils.selection import get_window_bins
    from utils.selection import get_neuron_mask
    from config import ANALYSIS_OPTIONS
    from data.load_responses import load_psth_mean
    from utils.smoothing import causal_boxcar

    # PSTH means are at sp_bin_width
    window_bins = get_window_bins(cd_ops, ANALYSIS_OPTIONS['sp_bin_width'])

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
        neuron_mask = get_neuron_mask(sess_dir, area, unit_filter)

        session = Session.load(str(sess_dir / 'session.pkl'))
        cids = session.unit_info['cluster_id'].values[neuron_mask]

        # load all conditions for this session first
        sess_means = {}
        sess_complete = True
        for event_type, conditions in [('tf', tf_conditions),
                                        ('lick', lick_conditions),
                                        ('blOn', bl_conditions)]:
            for cond in conditions:
                mean, _, t_ax = load_psth_mean(psth_path, event_type, cond,
                                               baseline_subtract=False)
                if mean is None:
                    sess_complete = False
                    break
                key = f'{event_type}/{cond}'
                t_axes[event_type] = t_ax
                masked = mean[neuron_mask]
                smoothed = causal_boxcar(masked, window_bins, axis=-1)
                sess_means[key] = smoothed
            if not sess_complete:
                break

        if not sess_complete:
            continue

        # session has all conditions - include its neurons
        unit_ids.extend([(sess_name, int(c)) for c in cids])
        for key, smoothed in sess_means.items():
            means.setdefault(key, []).append(smoothed)

    # concatenate across sessions (n_neurons dimension)
    concat_means = {}
    for key, session_means in means.items():
        concat_means[key] = np.concatenate(session_means, axis=0)

    return concat_means, t_axes, unit_ids


def calculate_tf_motor_alignment(npx_dir=PATHS['npx_dir_local'],
                                 cd_ops=CODING_DIM_OPS,
                                 area: str | None = None,
                                 unit_filter: list[str] | None = None):
    """
    compare TF and motor coding directions: cosine similarity between all pairs per
    block. also computes TF response projections onto motor dimensions for plotting.
    """
    save_dir = Path(npx_dir) / 'coding_dims'
    suffix = file_suffix(area, unit_filter)

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

        if len(shared) < cd_ops.get('min_neurons', 5):
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
        print(f'  {animal}: {n_shared}/{n_tf} TF, {n_shared}/{n_motor} motor neurons shared')

        dim_names = sorted(set(tf_r['dimensions']) & set(motor_r['dimensions']))

        # cosine similarity between all TF x motor dimension pairs (shared neurons)
        alignment = {v: {} for v in dim_names}
        for dim_name in dim_names:
            for block in ['early', 'late']:
                tf_dims = tf_r['dimensions'][dim_name].get(block, {})
                motor_dims = motor_r['dimensions'][dim_name].get(block, {})
                alignment[dim_name][block] = {}
                for tf_wl, tf_w in tf_dims.items():
                    tf_sub = tf_w[tf_idx]
                    for motor_wl, motor_w in motor_dims.items():
                        motor_sub = motor_w[motor_idx]
                        key = f'tf_{tf_wl}_x_motor_{motor_wl}'
                        alignment[dim_name][block][key] = cosine_similarity(tf_sub, motor_sub)

        # project TF responses onto motor dimensions (shared neurons only)
        shared_sessions = sorted(set(tf_r['included_sessions']) &
                                  set(motor_r['included_sessions']))
        mean_resps, t_axes, resp_ids = _load_mean_responses(
            animal, shared_sessions, npx_dir, area, unit_filter, cd_ops)

        # reorder resp rows to match motor_idx neuron order
        resp_id_to_idx = {uid: i for i, uid in enumerate(resp_ids)}

        # only keep neurons present in all three: tf, motor, and response data
        keep = [uid for uid in tf_shared_order if uid in resp_id_to_idx]
        resp_reorder = np.array([resp_id_to_idx[uid] for uid in keep])
        motor_reorder = np.array([motor_id_to_idx[uid] for uid in keep])

        tf_onto_motor = {v: {} for v in dim_names}
        for dim_name in dim_names:
            for block in ['early', 'late']:
                motor_dims = motor_r['dimensions'][dim_name].get(block, {})
                tf_onto_motor[dim_name][block] = {}
                for motor_wl, motor_w in motor_dims.items():
                    motor_sub = motor_w[motor_reorder]
                    tf_onto_motor[dim_name][block][motor_wl] = {}
                    for cond_key, resp in mean_resps.items():
                        if cond_key.startswith('tf/'):
                            resp_sub = resp[resp_reorder]
                            tf_onto_motor[dim_name][block][motor_wl][cond_key] = motor_sub @ resp_sub

        all_results[animal] = dict(
            alignment=alignment,
            tf_onto_motor=tf_onto_motor,
            tf_t_ax=t_axes.get('tf'),
        )

    out_path = save_dir / f'alignment_cd_{suffix}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved alignment to {out_path}')

    return all_results


#%% cross-dimension analysis

def cross_dimension_cosines(npx_dir=PATHS['npx_dir_local'],
                            cd_ops=CODING_DIM_OPS,
                            area: str | None = None,
                            unit_filter: list[str] | None = None,
                            n_perm: int = 500):
    """
    pairwise cosine similarities between block, tf, and motor coding dimensions, with permutation p-values. neurons
    are matched by unit_ids intersection
    """
    save_dir = Path(npx_dir) / 'coding_dims'
    suffix = file_suffix(area, unit_filter)

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
        if len(shared) < cd_ops.get('min_neurons', 5):
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

        dim_names = sorted(set(block_r['dimensions']) &
                          set(tf_r['dimensions']) &
                          set(motor_r['dimensions']))

        per_dim_name = {}
        for dim_name in dim_names:
            dims = {}
            for win_label, w in block_r['dimensions'][dim_name].items():
                dims[f'block_{win_label}'] = w[block_idx]
            for block in ['early', 'late']:
                for win_label, w in tf_r['dimensions'][dim_name].get(block, {}).items():
                    dims[f'tf_{block}_{win_label}'] = w[tf_idx]
                for win_label, w in motor_r['dimensions'][dim_name].get(block, {}).items():
                    dims[f'motor_{block}_{win_label}'] = w[motor_idx]

            names = sorted(dims.keys())
            n_dims = len(names)

            cosine_matrix = np.full((n_dims, n_dims), np.nan)
            p_matrix = np.full((n_dims, n_dims), np.nan)

            for i in range(n_dims):
                for j in range(i, n_dims):
                    vi = dims[names[i]]
                    vj = dims[names[j]]
                    real = cosine_similarity(vi, vj)
                    cosine_matrix[i, j] = real
                    cosine_matrix[j, i] = real

                    if i == j:
                        p_matrix[i, j] = 0.0
                        continue

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

            per_dim_name[dim_name] = dict(
                cosine_matrix=cosine_matrix,
                p_matrix=p_matrix,
                dim_names=names,
            )

        all_results[animal] = dict(per_dim_name=per_dim_name,
                                    n_shared=len(shared))
        print(f'  {animal}: {len(dim_names)} dim_names, {len(shared)} shared neurons')

    out_path = save_dir / f'cross_dimension_cosines_cd_{suffix}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved cross-dimension cosines to {out_path}')

    return all_results


def cross_dimension_projections(npx_dir=PATHS['npx_dir_local'],
                                cd_ops=CODING_DIM_OPS,
                                area: str | None = None,
                                unit_filter: list[str] | None = None,
                                ):
    """
    project event-aligned mean responses onto all block, tf, motor coding dimensions.
    neurons matched by unit_ids intersection across all three types
    """
    save_dir = Path(npx_dir) / 'coding_dims'
    suffix = file_suffix(area, unit_filter)

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
        if len(shared) < cd_ops.get('min_neurons', 5):
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

        dim_names = sorted(set(block_r['dimensions']) &
                           set(tf_r['dimensions']) &
                           set(motor_r['dimensions']))

        # load mean responses and match to shared neurons
        mean_resps, t_axes, resp_ids = _load_mean_responses(
            animal, shared_sessions, npx_dir, area, unit_filter, cd_ops)

        resp_id_to_idx = {uid: i for i, uid in enumerate(resp_ids)}
        keep = [uid for uid in shared_order if uid in resp_id_to_idx]

        if len(keep) < cd_ops.get('min_neurons', 5):
            print(f'  {animal}: only {len(keep)} neurons in responses - skipping')
            continue

        resp_reorder = np.array([resp_id_to_idx[uid] for uid in keep])
        keep_set = set(keep)
        keep_in_shared = np.array([i for i, uid in enumerate(shared_order)
                                   if uid in keep_set])

        per_dim_name = {}
        for dim_name in dim_names:
            dims = {}
            for win_label, w in block_r['dimensions'][dim_name].items():
                dims[f'block_{win_label}'] = w[block_idx]
            for block in ['early', 'late']:
                for win_label, w in tf_r['dimensions'][dim_name].get(block, {}).items():
                    dims[f'tf_{block}_{win_label}'] = w[tf_idx]
                for win_label, w in motor_r['dimensions'][dim_name].get(block, {}).items():
                    dims[f'motor_{block}_{win_label}'] = w[motor_idx]

            projections = {}
            for dim_key, w in dims.items():
                w_sub = w[keep_in_shared]
                projections[dim_key] = {}
                for resp_key, resp in mean_resps.items():
                    resp_sub = resp[resp_reorder]
                    projections[dim_key][resp_key] = w_sub @ resp_sub

            per_dim_name[dim_name] = dict(
                projections=projections,
                dim_names=sorted(dims.keys()),
            )

        all_results[animal] = dict(
            per_dim_name=per_dim_name,
            t_axes=t_axes,
            n_shared=len(keep),
        )
        print(f'  {animal}: projected {len(mean_resps)} responses onto '
              f'{len(dim_names)} dim_names x dimensions ({len(keep)} neurons)')

    out_path = save_dir / f'cross_dimension_projections_cd_{suffix}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved cross-dimension projections to {out_path}')

    return all_results


#%% run all combos

def run_all_coding_dim_analyses(npx_dir=PATHS['npx_dir_local'],
                                cd_ops=CODING_DIM_OPS,
                                dim_name: str = 'cd',
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
        dict(area='hippocampus', unit_filter=None),3
    ]

    all_stats = {}
    for combo in combos:
        suffix = file_suffix(**combo)
        print(f'\n{"=" * 60}')
        print(f'  {suffix}')
        print(f'{"=" * 60}')

        combo_save = Path(save_dir) / suffix if save_dir else None

        for dim_type, extract_fn in [('tf', extract_tf_dimensions),
                                      ('motor', extract_motor_dimensions)]:
            extract_fn(npx_dir=npx_dir, cd_ops=cd_ops,
                       n_jobs=6, **combo)

        calculate_tf_motor_alignment(npx_dir=npx_dir, cd_ops=cd_ops, **combo)

        for dim_type in ['tf', 'motor']:
            stats = analyse_coding_dimensions(
                dim_type, npx_dir=npx_dir, cd_ops=cd_ops, dim_name=dim_name,
                save_dir=str(combo_save) if combo_save else None, **combo)
            all_stats[(suffix, dim_type)] = stats

        plot_alignment(npx_dir=npx_dir, dim_name=dim_name,
                       save_dir=str(combo_save) if combo_save else None,
                       **combo)

    return all_stats
