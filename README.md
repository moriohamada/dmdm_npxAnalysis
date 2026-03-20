# dmdm neuropixels analysis

(In progress) analysis pipeline for dmdm dataset (Khilkevich & Lohse et al),
focussing on effects of behavioural and neural correlates of temporal expectation (early/late hazard-rate blocks).

## Directory structure

```
data/                shared data loading: sessions, FR matrices, event timings, PSTHs
utils/               filtering, smoothing, downsampling, normalisation, file I/O, ROI definitions
lick_pred/           lick prediction model: features, training, analysis, plotting, HPC scripts
glm/                 Poisson GLM per neuron: fitting, classification, kernel plots, HPC scripts
single_unit/         unit preferences (TF selectivity, block/lick modulation) and PSTH plots
population/          PCA, linear dynamical systems, empirical flow fields, trajectory plots
demixing/            SAE and causal LFADS for latent factor extraction from neural data
behaviour/           behavioural analyses: psychometrics, ELTA/ELTC, hazard rates, leaky integrator
config.py            all paths, analysis parameters, plot options
testing_grounds.py   neural analysis runner script (run cells top to bottom)
behavioural_analysis.py  behavioural analysis runner script
```

## Pipeline

### Neural analysis (`testing_grounds.py`)

1. Preprocessing raw data files to extract session data, neural responses
2. Behavioural lick prediction model (logistic regression and neural network)
3. Poisson GLM per neuron (stimulus + behaviour kernels, lesion analysis)
4. Unit preferences (TF selectivity, block modulation, lick modulation)
5. Downsample FR matrices to 50ms bins for population analyses
6. PCA on event-aligned population responses
7. Linear dynamical systems / empirical flow fields in PC space
8. Sparse autoencoder (SAE) or causal LFADS for latent factor demixing

### Behavioural analysis (`behavioural_analysis.py`)

1. Build per-subject trial DataFrames from npx Session objects
2. Psychometric and chronometric functions
3. Lick-triggered stimulus averages (ELTA) and covariance structure (ELTC via PCA)
4. FA hazard rates by block
5. Pulse-aligned lick probability
6. Leaky integrator model (grid search per subject per block, HPC parallelised)

## Key data notes

- FR matrices are z-scored per unit at creation time (`data/load_npx.py`). Don't z-score again for PSTHs or projections
- Original FR bin width is 10ms. Population analyses use 50ms bins via downsampled parquet files
- PSTHs are stored as (nEv x nN x nT) in HDF5. Means are (nN x nT)
- Brain region groupings are defined in `utils/rois.py`
- Session objects are saved slimmed (no FR matrix or raw neural data) as pickle files
- Block colours (orange/purple) are defined once in `config.py` `PLOT_OPTIONS` and shared across all plotting code
