"""
walk sessions and classify units for a given glm fit_type
"""
import os
import argparse
import pandas as pd
from pathlib import Path
from config import PATHS, GLM_OPTIONS
from neuron_prediction.results import FIT_TYPES
from neuron_prediction.results.classify import (
    classify_units, classify_units_perblock,
)


def _print_counts(df, group_names, label):
    counts = {g: int(df[f'{g}_sig'].sum()) if f'{g}_sig' in df else 0
              for g in group_names}
    parts = ', '.join(f'{v} {k}' for k, v in counts.items())
    print(f'  {label}: {len(df)} neurons: {parts}')


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

            if fit_type == 'glm_perblock':
                classify_units_perblock(sess_dir, fit_type=fit_type)
                for block in ('early', 'late'):
                    csv = Path(sess_dir) / f'{fit_type}_classifications_{block}.csv'
                    if csv.exists():
                        _print_counts(pd.read_csv(csv), group_names, block)
            else:
                df = classify_units(sess_dir, fit_type)
                _print_counts(df, group_names, fit_type)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit-type', required=True, choices=FIT_TYPES)
    args = ap.parse_args()
    collect(args.fit_type)
