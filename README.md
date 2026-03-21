# dmdm neuropixels analysis

(In progress) analysis pipeline for dmdm dataset (Khilkevich & Lohse et al),
focussing on effects of behavioural and neural correlates of temporal expectation (early/late hazard-rate blocks).

## Directory structure

```
data/                    data loading/handling: sessions, FR matrices, event timings, PSTHs
utils/                   shared utility functions
behaviour/               behavioural analyses
lick_pred/               lick prediction model
glm/                     Poisson GLM 
single_unit/             unit preferences (TF selectivity, block/lick modulation) and PSTH plots
population/              PCA, lds, population plots
demixing/                SAE and causal LFADS
config.py                all paths, analysis parameters, plot options
neural_analysis.py       draft analysis pipeline
behavioural_analysis.py  behavioural analysis runner script
```

## Pipeline

### Behavioural analysis (`behavioural_analysis.py`)

1. Build per-subject trial DataFrames from npx Session objects
2. Psychometric and chronometric functions
3. Lick-triggered stimulus averages and covariance structure 
4. FA hazard rates by block
5. Pulse-aligned lick probability
6. Leaky integrator model (grid search per subject per block, run on hpc)

### Neural analysis (for now: `neural_analysis.py`)

1. Preprocessing
2. Behavioural lick prediction model
3. Poisson GLM per neuron
4. Unit preferences
5. Downsample FR matrices for population analyses
6. PCA on event-aligned population responses
7. LDS
8. Latent factor analysis w SAE/modified LFADS


#### Data notes

- FR matrices are z-scored per unit at creation time (`data/load_npx.py`). Don't z-score again for PSTHs or projections
- Original FR bin width is 10ms. Population analyses use 50ms bins (separate _ds parquet files)
- PSTHs are stored as (nEv x nN x nT) in hdf5. Means are (nN x nT)
- Brain region groupings defined in `utils/rois.py`
- Session objects are saved in pickle files without FR matrix, raw neural data, or raw daq lines
