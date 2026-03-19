# dmdm neuropixels analysis

(In progress) neuropixels analysis pipeline for dmdm dataset (Khilkevich & Lohse et al),
focussing on effects of behavioural and neural correlates of temporal expectation (early/late hazard-rate blocks).

## Directory structure

```
data/                shared data loading: sessions, FR matrices, event timings, PSTHs
utils/               filtering, smoothing, downsampling, normalisation, file I/O, ROI definitions
lick_pred/           lick prediction model: features, training, analysis, plotting, HPC scripts
glm/                 Poisson GLM per neuron: fitting, classification, kernel plots, HPC scripts
single_unit/         unit preferences (TF selectivity, block/lick modulation) and PSTH plots
population/          PCA, linear dynamical systems, empirical flow fields, trajectory plots
config.py            all paths, analysis parameters, plot options
testing_grounds.py   main runner script (run cells top to bottom)
```

## Pipeline

1. Preprocessing raw data files to extract session data, neural responses
2. Behavioural lick prediction model (logistic regression and neural network)
3. Poisson GLM per neuron (stimulus + behaviour kernels, lesion analysis)
4. Unit preferences (TF selectivity, block modulation, lick modulation)
5. Downsample FR matrices to 50ms bins for population analyses
6. PCA on event-aligned population responses
7. Linear dynamical systems / empirical flow fields in PC space

## Key data notes

- FR matrices are z-scored per unit at creation time (`data/load_npx.py`). Don't z-score again for PSTHs or projections
- Original FR bin width is 10ms. Population analyses use 50ms bins via downsampled parquet files
- PSTHs are stored as (nEv x nN x nT) in HDF5. Means are (nN x nT)
- Brain region groupings are defined in `utils/rois.py`
- Session objects are saved slimmed (no FR matrix or raw neural data) as pickle files
