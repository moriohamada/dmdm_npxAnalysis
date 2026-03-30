### dmdm: temporal expectation project

---
_**In progress**_

---
Analysis pipeline for the dmdm dataset (Khilkevich & Lohse et al). Brain-wide Neuropixels recordings in mice doing a visual change detection task with temporal expectation (early/late hazard-rate blocks).

Dataset comprises ~15k units, 51 regions, 15 mice, 114 sessions.

#### Structure

```
data/                    session class def, FR matrices, event timings, preprocessing
utils/                   shared utilities, brain region groupings

config.py                paths, analysis parameters, plot options
neural_analysis.py       main neural analysis runner
behavioural_analysis.py  main behavioural analysis runner

behaviour/               psychometrics, lick-triggered averages, FA hazard
behaviour/integrator/    leaky integrator model (grid search on HPC)
lick_pred/               lick prediction from neural activity
neuron_prediction/       single-unit prediction models
  glm/                   poisson GLM with group lasso
  network/               poisson networks (linear + hidden layer)
single_unit/             unit preferences, PSTH plots
population/              PCA, LDS, population trajectories
demixing/                SAE and causal LFADS
tuning_curves/           single-unit TF tuning by block
coding_dims/             coding dimension rotation, motor subspace projection
```

### Behavioural analyses


**Basic analyses** (`behaviour/`) - per-subject trial DataFrames from Session objects. Psychometric/chronometric functions, lick-triggered stimulus averages and covariance, FA hazard rates by block, pulse-aligned lick probability.

**Leaky integrator** (`behaviour/integrator/`) - grid search over time constant and threshold per subject per block. Runs on HPC.

**Lick prediction** (`lick_pred/`) - predict lick timing from neural population activity. Leave-one-session-out CV.

### Neural analyses

**Preprocessing** (`data/`) - extract FR matrices and event timings from raw data on ceph, downsample to 50ms bins for population analyses.

**Single unit responses** (`single_unit/`) - Per-neuron PSTH plots & preference index calculation

**Poisson GLM** (`neuron_prediction/glm/`) - per-neuron GLM with design matrix containing TF, events, lick 
preparation/execution, time ramp, block, and motion signals. Time-shifted predictor kernels, group lasso regularisation, lesion analysis for unit classification. Runs on HPC via SLURM array jobs. Config: `GLM_OPTIONS`.

**Nonlinear fits** (`neuron_prediction/network/`) - same design matrix, Poisson networks with one hidden ReLU layer and orthogonality penalty. Inner CV per hidden size for regularisation selection, linear baseline for comparison. Same lesion framework as GLM. Config: `NETWORK_OPTIONS`.

**Population** (`population/`) - PCA on event-aligned responses, LDS.

**Demixing** (`demixing/`) - SAE and causal LFADS for learning interpretable latent factors from neural activity.

**Tuning curves** (`tuning_curves/`) - single-unit TF tuning by block. OLS fit of firing rate vs TF per block, permutation tests for gain significance. TF-responsive = significant gain in either block at p < 0.025. Quantile-binned curves with SEM. Results per session. Config: `TUNING_CURVE_OPS`.

**Coding dimensions** (`coding_dims/`) - two analyses comparing TF coding between early/late blocks. (1) Coding dimension rotation: time-resolved TF coding vector per block, between-block cosine similarity over post-pulse time, null from block-label shuffling. (2) Motor dimension projection: PCA motor subspace from lick-aligned activity (even/odd CV, fake-lick null), TF responses projected onto motor and non-motor dimensions per block. Results per animal. Config: `CODING_DIM_OPS`.

---

##### Data notes

- FR matrices are z-scored per unit at creation time. Don't z-score again downstream
- Original FR bins are 10ms. Population analyses use 50ms (separate `_ds` parquet files)
- PSTHs stored as (nEv x nN x nT) in HDF5. Means are (nN x nT)
- Brain region groupings in `utils/rois.py`
- Session objects saved without FR matrix, raw neural data, or raw DAQ lines
- GLM design matrices saved per session (`glm_counts.npy`, `glm_design.npy`, `glm_spec.pkl`). GLM and network results saved per neuron in `glm_results/` and `network_results/` subdirectories
