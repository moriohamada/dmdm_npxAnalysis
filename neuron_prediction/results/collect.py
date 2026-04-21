"""
walk sessions and classify units for a given glm fit_type
"""
import os
import argparse
from config import PATHS, GLM_OPTIONS
from neuron_prediction.results import FIT_TYPES
from neuron_prediction.results.classify import classify_units


def collect(fit_type, npx_dir=None):
    """classify units across all sessions that have <fit_type>_results/"""
    if npx_dir is None:
        npx_dir = PATHS['npx_dir_local']
    group_names = list(GLM_OPTIONS['lesion_groups'].keys())

    for subj in sorted(os.listdir(npx_dir)):
        subj_dir = os.path.join(npx_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for sess in sorted(os.listdir(subj_dir)):
            sess_dir = os.path.join(subj_dir, sess)
            results_dir = os.path.join(sess_dir, f'{fit_type}_results')
            if not os.path.isdir(results_dir):
                continue
            print(f'{subj}/{sess}')
            df = classify_units(sess_dir, fit_type)
            counts = {g: df[f'{g}_sig'].sum() if f'{g}_sig' in df else 0
                      for g in group_names}
            parts = ', '.join(f'{v} {k}' for k, v in counts.items())
            print(f'  {len(df)} neurons: {parts}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit-type', required=True, choices=FIT_TYPES)
    args = ap.parse_args()
    collect(args.fit_type)
