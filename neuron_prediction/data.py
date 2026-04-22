"""shared data loading and fold assignment for neuron prediction models"""
import hashlib
import numpy as np
import pandas as pd
import pickle
from pathlib import Path


def analysis_trials(session):
    """trials participating in GLM analyses - excludes trialoutcome=='Ref'"""
    tr = session.trials
    if 'trialoutcome' in tr.columns:
        return tr[tr['trialoutcome'] != 'Ref']
    return tr


def lick_times(session):
    """unified lick time for Hit and FA trials

    cascade: motion_onset -> first_lick - delta -> RT - delta
    delta = per-session median(first_lick - motion_onset) where both exist
    """
    tr = analysis_trials(session)
    sub = tr[(tr['IsHit'] == 1) | (tr['IsFA'] == 1)].copy()

    both = sub['motion_onset'].notna() & sub['first_lick'].notna()
    delta = (sub.loc[both, 'first_lick'] - sub.loc[both, 'motion_onset']).median()
    if pd.isna(delta):
        delta = 0.0

    times = sub['motion_onset'].copy()

    need = times.isna() & sub['first_lick'].notna()
    times.loc[need] = sub.loc[need, 'first_lick'] - delta

    if 'rt_FA' in sub.columns:
        need = times.isna() & (sub['IsFA'] == 1) & sub['rt_FA'].notna()
        times.loc[need] = sub.loc[need, 'Baseline_ON_rise'] + sub.loc[need, 'rt_FA'] - delta

    if 'rt_RT' in sub.columns:
        need = times.isna() & (sub['IsHit'] == 1) & sub['rt_RT'].notna()
        times.loc[need] = sub.loc[need, 'Change_ON_rise'] + sub.loc[need, 'rt_RT'] - delta

    return times.dropna().values


def load_glm_inputs(sess_dir):
    """load prepped GLM data: counts, design matrix, col_map, time axis, valid mask"""
    sess_dir = Path(sess_dir)
    counts = np.load(sess_dir / 'glm_counts.npy')
    X = np.load(sess_dir / 'glm_design.npy')
    t_ax = np.load(sess_dir / 'glm_t_ax.npy')
    valid_mask = np.load(sess_dir / 'glm_valid.npy')
    with open(sess_dir / 'glm_spec.pkl', 'rb') as f:
        col_map = pickle.load(f)
    return counts, X, col_map, t_ax, valid_mask


CONTINUOUS_PREDICTORS = {'tf', 'time_ramp'}


def normalise_design_matrix(X_train, X_test, col_map):
    """z-score continuous predictor columns, leave binary/one-hot as-is

    only normalises columns belonging to predictor groups listed in
    CONTINUOUS_PREDICTORS. all others (event impulses, phase one-hots,
    block indicator) are left untouched.
    """
    mu = np.zeros(X_train.shape[1])
    sd = np.ones(X_train.shape[1])

    for name, (col_slice, _) in col_map.items():
        if name not in CONTINUOUS_PREDICTORS:
            continue
        mu[col_slice] = X_train[:, col_slice].mean(axis=0)
        sd[col_slice] = X_train[:, col_slice].std(axis=0)

    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def get_fold_indices(n_samples, n_folds, seed):
    """deterministic fold assignment for cross-validation

    uses a per-neuron seed so both GLM and network get identical splits
    """
    rng = np.random.RandomState(seed)
    fold_ids = np.tile(np.arange(n_folds), n_samples // n_folds + 1)[:n_samples]
    rng.shuffle(fold_ids)
    return fold_ids


def get_trial_fold_indices(trials_df, t_ax, n_folds, seed,
                           ignore_first_n=0,
                           exclude_outcomes=('Ref',)):
    """assign CV folds at trial level, map back to time bins

    all bins within a trial share the same fold. bins outside valid
    trials get fold_id = -1. skips transition trials (tr_in_block
    <= ignore_first_n) and trials whose trialoutcome is in
    exclude_outcomes (Ref by default).

    returns (T,) int array of fold IDs
    """
    rng = np.random.RandomState(seed)
    excluded = set(exclude_outcomes or ())

    # collect valid trial boundaries
    trial_bounds = []
    for _, row in trials_df.iterrows():
        if ignore_first_n > 0 and row['tr_in_block'] <= ignore_first_n:
            continue
        if excluded and row.get('trialoutcome') in excluded:
            continue
        bl_on = row['Baseline_ON_rise']
        tr_end = np.nanmax([row['Baseline_ON_fall'],
                            row.get('Change_ON_fall', np.nan)])
        if np.isnan(bl_on) or np.isnan(tr_end):
            continue
        trial_bounds.append((bl_on, tr_end))

    n_trials = len(trial_bounds)
    if n_trials == 0:
        return np.full(len(t_ax), -1, dtype=int)

    # assign folds to trials
    trial_folds = np.tile(np.arange(n_folds),
                          n_trials // n_folds + 1)[:n_trials]
    rng.shuffle(trial_folds)

    # map to time bins
    fold_ids = np.full(len(t_ax), -1, dtype=int)
    for i, (bl_on, tr_end) in enumerate(trial_bounds):
        mask = (t_ax >= bl_on) & (t_ax < tr_end)
        fold_ids[mask] = trial_folds[i]

    return fold_ids


def neuron_seed(sess_dir, neuron_idx):
    """reproducible seed from session path + neuron index

    uses hashlib (not hash()) so the seed is stable across python processes
    """
    key = f'{sess_dir}_{neuron_idx}'.encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % (2**31)


def select_neurons(sess_dir, min_r=0.2, require_tf=False):
    """filter neurons by GLM classification

    returns list of neuron indices passing criteria
    """
    import pandas as pd
    class_path = Path(sess_dir) / 'glm_classifications.csv'
    if not class_path.exists():
        return []
    df = pd.read_csv(class_path)
    mask = df['mean_r'] >= min_r
    if require_tf:
        mask &= df['tf_sig'] == True
    return df.loc[mask, 'neuron_idx'].tolist()


def convert_job_map_to_hpc(csv_path):
    """rewrite sess_dir paths in a job map CSV from local to ceph"""
    import pandas as pd
    from config import LOCAL_PATHS, HPC_PATHS
    df = pd.read_csv(csv_path)
    df['sess_dir'] = df['sess_dir'].str.replace(
        LOCAL_PATHS['npx_dir'], HPC_PATHS['npx_dir'])
    df.to_csv(csv_path, index=False)
    print(f'Converted {len(df)} rows in {csv_path}')
    return df


def build_network_job_map(npx_dir=None, output_path=None):
    """build CSV mapping SLURM array index -> (session_dir, neuron_index)

    includes all neurons with prepped GLM design matrices
    """
    import os
    import pandas as pd
    from config import PATHS
    from data.session import Session

    if npx_dir is None:
        npx_dir = PATHS['npx_dir_local']
    if output_path is None:
        output_path = os.path.join(npx_dir, 'network_job_map.csv')

    rows = []
    for subj in sorted(os.listdir(npx_dir)):
        subj_dir = os.path.join(npx_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for sess in sorted(os.listdir(subj_dir)):
            sess_dir = os.path.join(subj_dir, sess)
            if not os.path.exists(os.path.join(sess_dir, 'glm_counts.npy')):
                continue

            sess_data = Session.load(os.path.join(sess_dir, 'session.pkl'))
            cluster_ids = sess_data.fr_stats.index.values

            for i in range(len(sess_data.fr_stats)):
                rows.append({
                    'job_idx': len(rows),
                    'sess_dir': sess_dir,
                    'neuron_idx': i,
                    'cluster_id': cluster_ids[i],
                    'animal': sess_data.animal,
                    'session': sess_data.name,
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f'Network job map: {len(df)} neurons across '
          f'{df["sess_dir"].nunique() if len(df) else 0} sessions')
    print(f'Saved to {output_path}')
    return df
