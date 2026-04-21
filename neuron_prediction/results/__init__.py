"""
shared post-fit utilities for glm variants (glm, glm_ridge, glm_unreg):
unit classification, kernel extraction, per-neuron and population plots

naming convention per fit_type:
    <sess_dir>/<fit_type>_results/neuron_{i}.npz
    <sess_dir>/<fit_type>_classifications.csv
    <plots_dir>/<animal>/<session>/<fit_type>_kernels/
"""

FIT_TYPES = ('glm', 'glm_ridge', 'glm_unreg')
