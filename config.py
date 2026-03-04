
PATHS = dict(
    npx_dir_ceph = '/mnt/ceph/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx',
    npx_dir_local = '/media/morio/Data_Fast/dmdm_temporalExpectation/npx/',
)

ANALYSIS_OPTIONS = dict(
    spBinWidth     = 10/1000,    # s
    spSmoothWidth  = 50/1000,    # s; size of casual boxcar fitler

    tfOutlier = 1.0,             # std deviations away from mean to consider an outlier

    minTrialDur   = 2.0,         # s; minimum trial duration
    rmvTimeAround = 1.5,         # s; remove this time around trial,

    minHitsInSession = 30,       # trials; ignore sessions with fewer hits than this
    ignoreFirstBlockTrials = 5,  # trials; ignroe first n trials after a new block

    trSplitTime = 8,             # s; for splitting events into early vs late in trial
)

