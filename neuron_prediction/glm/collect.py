"""
post-hoc script: collect per-neuron GLM results and classify units across all sessions
"""
import os
from config import PATHS, GLM_OPTIONS
from neuron_prediction.glm.fit import classify_units


if __name__ == '__main__':
    npx_dir = PATHS['npx_dir_local']
    group_names = list(GLM_OPTIONS['lesion_groups'].keys())

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
            counts = {g: df[f'{g}_sig'].sum() if f'{g}_sig' in df else 0
                      for g in group_names}
            parts = ', '.join(f'{v} {k}' for k, v in counts.items())
            print(f'  {len(df)} neurons: {parts}')
