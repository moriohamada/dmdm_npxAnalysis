"""
post-hoc script: collect per-neuron GLM results and classify units across all sessions
"""
import os
from config import PATHS
from analyses.glm import classify_units

npx_dir = PATHS['npx_dir_local']

for subj in sorted(os.listdir(npx_dir)):
    subj_dir = os.path.join(npx_dir, subj)
    if not os.path.isdir(subj_dir):
        continue
    for sess in sorted(os.listdir(subj_dir)):
        sess_dir = os.path.join(subj_dir, sess)
        results_dir = os.path.join(sess_dir, 'glm_results')
        if not os.path.isdir(results_dir):
            continue
        print(f'{subj}/{sess}')
        df = classify_units(sess_dir)
        n_tf = df['tf_sig'].sum() if 'tf_sig' in df else 0
        n_lp = df['lick_prep_sig'].sum() if 'lick_prep_sig' in df else 0
        n_le = df['lick_exec_sig'].sum() if 'lick_exec_sig' in df else 0
        print(f'  {len(df)} neurons: {n_tf} TF-resp, {n_lp} lick-prep, {n_le} lick-exec')
