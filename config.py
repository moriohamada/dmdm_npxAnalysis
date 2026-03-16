
PATHS = dict(
    npx_dir_ceph  = '/mnt/ceph/public/projects/MoHa_20260212_dmdmTemporalExpectation'
                    '/data/npx_converted',
    npx_dir_local = '/media/morio/Data_Fast/dmdm_temporalExpectation/npx/',
    plots_dir     = '/media/morio/Data_Fast/dmdm_temporalExpectation/plots/',
    pref_dir      = '/media/morio/Data_Fast/dmdm_temporalExpectation/preferences/',
)

ANALYSIS_OPTIONS = dict(
    sp_bin_width     =10 / 1000,    # s
    sp_smooth_width  =50 / 1000,    # s; size of casual boxcar fitler

    tf_outlier = 1.0,             # std deviations away from mean to consider an outlier

    min_trial_dur   = 2.0,         # s; minimum trial duration
    rmv_time_around = 1.5,         # s; remove this time around trial,

    min_hits_in_session = 30,       # trials; ignore sessions with fewer hits than this
    ignore_first_trials_in_block = 5,  # trials; ignroe first n trials after a new block

    tr_split_time = 8,             # s; for splitting events into early vs late in trial

    # Response timings
    tf_resp_win = (0.1, 0.5),
    tf_context = (-0.5, -0.1),
    lick_bl    = (-2, -1),
    lick_pre   = (-0.4, 0),

    n_iter = 1000,                # default num shuffles
    alpha  = .05,                 # default signficance threshold

    # Unit filtering
    min_fr    = 1.0,              # Hz; exclude units with mean FR below this
    min_fr_sd = 0.5,              # Hz; exclude units with FR std below this

    # Population analysis
    pop_bin_width = 50 / 1000,    # s; bin width for population analyses - for limiting ram load
    n_pcs = 10,                   # number of PCs to extract

    # Dynamical systems
    lds_n_folds = 5,              # k-fold CV for LDS fitting

)

PLOT_COLOURS = dict(
    block = {
        'early': (230/255, 97/255,   1/255),   # orange
        'late':  ( 94/255, 60/255, 153/255),   # purple
    },

    tf = {
        'pos': (204/255,  37/255,  41/255),  # Red
        'neg': ( 57/255, 106/255, 177/255),  # Blue
    },

    ch_tf_cmap = 'viridis',

)
