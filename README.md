### dmdm: temporal expectation project

---
_**In progress**_

---
Analysis pipeline for the dmdm dataset (Khilkevich & Lohse et al). Brain-wide Neuropixels recordings in mice doing a visual change detection task with temporal expectation (early/late hazard-rate blocks).

Dataset comprises ~15k units, 51 regions, 15 mice, 114 sessions.

#### Structure

```
config.py                paths, analysis parameters, plot options
neural_analysis.py       main neural analysis runner
behavioural_analysis.py  main behavioural analysis runner

data/                    session class def, FR matrices, event timings, preprocessing
utils/                   shared utilities, brain region groupings

behaviour/               psychometrics, lick-triggered averages, FA hazard
behaviour/integrator/    leaky integrator model  
lick_pred/               lick prediction task features and behavioural hsitory
neuron_prediction/       single-unit prediction models
  glm/                   poisson GLM with group lasso
  network/               ff networks to allow non-linear interactions
  hybrid/                hybrid model - linear skips to output + specified non-linear
single_unit/             unit preferences, PSTHs
population/              PCA, LDS, population trajectories
demixing/                SAE and causal LFADS
tuning_curves/           single-unit TF tuning by block
coding_dims/             coding dimension rotation, motor subspace projection
```

### Behavioural analyses


**Basic analyses** (`behaviour/`) - per-subject trial dataframes from Session objects. 
Block-dependent psychometric/chronometric, lick-triggered stimulus mean and covariance calculation, hazard rates, pulse-aligned lick probability.

**Leaky integrator** (`behaviour/integrator/`) - grid search over integration, gainm threshold per subject per block. 

**Lick prediction** (`lick_pred/`) - predict lick timing based on stimulus, trial history etc.

### Neural analyses

**Preprocessing** (`data/`) - extract FR matrices and event timings, downsample to 50ms bins for population analyses.

**Single unit responses** (`single_unit/`) - single neuron PSTH plots & preference index calculation

**Poisson GLM** (`neuron_prediction/glm/`) - per-neuron GLM with group lasso regularisation, lesion analysis for unit 
classification (TF/time/block/lick responsive). Config: `GLM_OPTIONS`.

**Nonlinear fits** (`neuron_prediction/network/`) - same design matrix, ff nets with one hidden ReLU layer. 
Shuffling tests for individual predictor groups + pairwise/three-way interaction shuffling to detect specified 
non-linear interactions (TF x block x time). Config: `NETWORK_OPTIONS`.

**Hybrid model** (`neuron_prediction/hybrid/`) - linear skip connection (all predictors to output) + masked hidden 
layer (configurable subset of predictor groups), to identify specific predictor interactions the network captures beyond the GLM.

**Population** (`population/`) - PCA on event-aligned responses, LDS.

**Demixing** (`demixing/`) - SAE and causal LFADS for learning interpretable latent factors from neural activity.

**Tuning curves** (`tuning_curves/`) - single unit TF tuning by block. Config: `TUNING_CURVE_OPS`.

**Coding dimensions** (`coding_dims/`) - two analyses comparing TF coding between early/late blocks. 
1: coding dimension rotations: time-resolved TF and lick coding vector per block, between-block cosine similarity over 
post-pulse time. 
2: dimension alignment - do motor/tf coding dims rotate relative to each other depending on expectation? Config: 
`CODING_DIM_OPS`.

---

##### Data notes

- FR matrices are z-scored per unit at creation time  
- Original FR bins are 10ms. Population analyses use 50ms (separate `_ds` parquet files)
- PSTHs stored as (nEv x nN x nT) in HDF5. Means are (nN x nT)
- Brain region groupings in `utils/rois.py`
- Session objects saved without FR matrix, raw neural data, or raw DAQ lines
- GLM design matrices saved per session (`glm_counts.npy`, `glm_design.npy`, `glm_spec.pkl`). 
