# dmdm neuropixels analysis

(In progress) neuropixels analysis pipeline for dmdm dataset (Khilkevich & Lohse et al., Nature 2024.)
analysing effects of temporal expectation (early/late hazard-rate blocks). 

## Directory structure

``` 
data/                loading raw data, building FR matrices, extracting event timings and PSTHs
analyses/            preferences, PCA, LDS, flow fields, lick prediction model
utils/               filtering, smoothing, downsampling, normalisation, file I/O, ROI definitions
visualisation/       plotting (PSTHs, preferences, PC-space trajectories, lick prediction)
config.py            all paths, analysis parameters, plot options
testing_grounds.py   main runner script (run cells top to bottom) - will be changed to analyse.py
``` 

## Pipeline

1. Raw neuropixels data (ceph) - FR matrices + PSTHs (local parquet/HDF5)
2. Behavioural lick prediction model (logistic regression and neural network)
3. Unit preferences (TF selectivity, block modulation, lick modulation)
4. Downsample FR matrices to 50ms bins for population analyses
5. PCA on event-aligned population responses
6. Linear dynamical systems / empirical flow fields in PC space

## Key data notes

- FR matrices are z-scored per unit at creation time (`data/load_npx.py`). Don't z-score again for PSTHs or projections
- Original FR bin width is 10ms. Population analyses use 50ms bins via downsampled parquet files
- PSTHs are stored as (nEv x nN x nT) in HDF5. Means are (nN x nT)
- Brain region groupings are defined in `utils/rois.py`
- Session objects are saved slimmed (no FR matrix or raw neural data) as pickle files
