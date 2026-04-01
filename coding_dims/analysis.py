"""coding dimension analysis: statistics, alignment, projections"""

import numpy as np
import pickle
from pathlib import Path

from config import PATHS, CODING_DIM_OPS
from coding_dims.extract import _file_suffix, cosine_similarity
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
            p = np.mean(valid_null >= real) if len(valid_null) > 0 else np.nan

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
    population-level significance test for between-block consistency.
    for each permutation: sample one null value per animal, average.
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
        p_value = np.mean(valid_null_means >= observed_mean) if len(valid_null_means) > 0 else np.nan

        out[wl] = dict(
            observed_mean=observed_mean,
            null_means=null_means,
            p_value=p_value,
            n_animals=len(real_vals),
        )
    return out


def analyse_coding_dimensions(dim_type, npx_dir=PATHS['npx_dir_local'],
                              bm_ops=CODING_DIM_OPS,
                              area=None, unit_filter=None,
                              save_dir=None):
    """
    main analysis runner for tf or motor coding dimensions.
    loads extracted results, computes per-animal and pooled significance, plots.
    dim_type: 'tf' or 'motor'
    """
    suffix = _file_suffix(area, unit_filter)
    results = load_dimension_results(dim_type, npx_dir, area, unit_filter)

    per_animal = per_animal_significance(results)
    pooled = pooled_null_test(results, n_perm=10000)

    for wl in sorted(per_animal.keys()):
        pa = per_animal[wl]
        po = pooled.get(wl, {})
        n_sig = np.sum(pa['p_values'] < 0.05)
        print(f'{dim_type} [{suffix}] {wl}: '
              f'{len(pa["animals"])} animals, '
              f'{n_sig}/{len(pa["animals"])} sig at p<0.05, '
              f'mean cosine={np.nanmean(pa["cosines"]):.3f}, '
              f'pooled p={po.get("p_value", np.nan):.4f}')

    from coding_dims.plotting import plot_tf_dimensions, plot_motor_dimensions
    plot_fn = plot_tf_dimensions if dim_type == 'tf' else plot_motor_dimensions
    plot_fn(npx_dir=npx_dir, save_dir=save_dir, area=area, unit_filter=unit_filter)

    return dict(per_animal=per_animal, pooled=pooled)


#%% tf-motor alignment

def _load_mean_responses(animal, included_sessions, npx_dir, area, unit_filter, bm_ops):
    """
    load smoothed, neuron-masked, session-concatenated mean responses for one animal.
    returns dict of {condition: (n_neurons, n_time)} and time axes per event type
    """
    from coding_dims.extract import _get_neuron_mask, _get_window_bins
    from data.load_responses import load_psth_mean
    from utils.smoothing import causal_boxcar
    from config import ANALYSIS_OPTIONS

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

    for sess_name in included_sessions:
        sess_dir = Path(npx_dir) / animal / sess_name
        psth_path = str(sess_dir / 'psths.h5')
        neuron_mask = _get_neuron_mask(sess_dir, area, unit_filter)

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

    return concat_means, t_axes


def calculate_tf_motor_alignment(npx_dir=PATHS['npx_dir_local'],
                                 bm_ops=CODING_DIM_OPS,
                                 area=None, unit_filter=None):
    """
    compare TF and motor coding directions: cosine similarity between all pairs,
    per block. also computes TF response projections onto motor dimensions for
    plotting.
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

        tf_sessions = tf_r.get('included_sessions', [])
        motor_sessions = motor_r.get('included_sessions', [])
        if tf_sessions != motor_sessions:
            print(f'  {animal}: session mismatch '
                  f'(TF: {len(tf_sessions)}, motor: {len(motor_sessions)}) - skipping')
            continue

        # cosine similarity between all TF x motor dimension pairs
        alignment = {}
        for block in ['early', 'late']:
            tf_dims = tf_r['dimensions'].get(block, {})
            motor_dims = motor_r['dimensions'].get(block, {})
            alignment[block] = {}
            for tf_wl, tf_w in tf_dims.items():
                for motor_wl, motor_w in motor_dims.items():
                    key = f'tf_{tf_wl}_x_motor_{motor_wl}'
                    alignment[block][key] = cosine_similarity(tf_w, motor_w)

        # project TF responses onto motor dimensions
        mean_resps, t_axes = _load_mean_responses(
            animal, tf_sessions, npx_dir, area, unit_filter, bm_ops)

        tf_onto_motor = {}
        for block in ['early', 'late']:
            motor_dims = motor_r['dimensions'].get(block, {})
            tf_onto_motor[block] = {}
            for motor_wl, motor_w in motor_dims.items():
                tf_onto_motor[block][motor_wl] = {}
                for cond_key, resp in mean_resps.items():
                    if cond_key.startswith('tf/'):
                        tf_onto_motor[block][motor_wl][cond_key] = motor_w @ resp

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
