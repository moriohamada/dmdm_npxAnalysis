
PATHS = dict(
    npx_dir_ceph  = '/mnt/ceph/public/projects/MoHa_20260212_dmdmTemporalExpectation'
                    '/data/npx_converted',
    npx_dir_local = '/media/morio/Data_Fast/dmdm_temporalExpectation/npx/',
    plots_dir     = '/media/morio/Data_Fast/dmdm_temporalExpectation/plots/',
    pref_dir      = '/media/morio/Data_Fast/dmdm_temporalExpectation/preferences/',
)

ANALYSIS_OPTIONS = dict(
    sp_bin_width     = 10 / 1000,     # s
    sp_smooth_width  = 50 / 1000,     # s; size of casual boxcar fitler

    tf_outlier = 1.0,                 # std deviations away from mean to consider an outlier

    min_trial_dur   = 2.0,            # s; minimum trial duration
    rmv_time_around = 1.5,            # s; remove this time around trial,

    min_hits_in_session = 30,         # trials; ignore sessions with fewer hits than this
    ignore_first_trials_in_block = 5, # trials; ignroe first n trials after a new block

    tr_split_time = 8,                # s; for splitting events into early vs late in trial

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
    lds_n_dims = 5,               # number of PCs for linear LDS
    lds_n_folds = 5,             # k-fold CV for LDS fitting
    flow_n_dims = 2,              # number of PCs for empirical flow field
    flow_n_bins = 15,             # bins per dimension for flow field
    flow_min_count = 50,          # minimum time bin pairs per grid bin to estimate flow

)

GLM_OPTIONS = dict(
    bin_width = 50 / 1000,  # s; matches TF pulse duration (3 frames at 60Hz)

    # predictor kernel windows (start, end) in seconds
    # positive lags = predictor precedes response, negative = follows
    kern_tf          = (-.25, 1.25),
    kern_trial_start = (0, 1.0),
    kern_change      = (0, 2.0),
    kern_lick_prep   = (-1.25, 0),
    kern_lick_exec   = (0, 0.5),
    kern_air_puff    = (0, 0.25),
    kern_reward      = (0, 0.4),
    kern_abort       = (-1.25, 0.25),
    kern_face_me     = (-0.05, 0.8),
    kern_running     = (-0.05, 0.8),
    kern_pupil       = (-0.75, 0.75),

    n_phase_bins = 12,

    # CV
    n_outer_folds = 10,
    n_inner_folds = 10,

    # unit classification thresholds
    min_r = 0.2,
    lesion_alpha = 0.01,

    # predictor groups to lesion together for unit classification
    # correlated predictors removed together to avoid compensation
    lesion_groups = {
        'tf':        ['tf', 'phase_up', 'phase_down'],
        'lick_prep': ['lick_prep'],
        'lick_exec': ['lick_exec', 'face_me', 'running'],
    },
)

LICK_PRED_OPS = dict(
    bin_width       = 50 / 1000,       # s; prediction bin width (matches pop_bin_width)
    tf_history_bins = 40,              # number of 50ms bins of TF history (2s)
    tf_subsample    = 3,               # subsample raw 60Hz TF by this factor

    # target
    lick_sigma_bins   = 2,             # gaussian kernel sigma in bins
    lick_extend_bins  = 5,             # extend trial 250ms past lick
    response_window   = 2.15,          # s; task response window after change onset

    # trial history
    max_time_since_reward = 300,       # s; cap for first trial before any reward

    # training
    hidden_sizes    = [8, 16, 32, 64],
    lambdas         = [0, 1e-4, 1e-3, 1e-2],
    lr              = 1e-4,
    batch_size      = 4096,
    max_epochs      = 800,
    sweep_epoch_frac = 0.25,           # fraction of max_epochs for quick sweep
    patience        = 50,             # early stopping patience (epochs)
    val_frac        = 0.1,            # fraction of training trials for early stopping
)

PLOT_OPTIONS = dict(
    colours = dict(
        block = {
            'early': (230/255, 97/255,   1/255),   # orange
            'late':  ( 94/255, 60/255, 153/255),   # purple
        },

        tf = {
            'pos': (204/255,  37/255,  41/255),  # Red
            'neg': ( 57/255, 106/255, 177/255),  # Blue
        },

        ch_tf_cmap = 'viridis',
    ),

    smooth_window_short = 250 / 1000,  # s; causal boxcar window for short trajectories (TF, blOn)
    smooth_window_long  = 500 / 1000,  # s; causal boxcar window for long trajectories
    # (bl, lick)
)
