import os
import numpy as np
from sys import platform


if platform == "linux" or platform == "linux2":
    LOCAL_PATHS = dict(
        npx_dir  = '/media/morio/Data_Fast/dmdm_temporalExpectation/npx/',
        ceph_dir = '/mnt/ceph/public/projects/MoHa_20260212_dmdmTemporalExpectation'
                   '/data/npx_converted',
        plots_dir = '/media/morio/Data_Fast/dmdm_temporalExpectation/plots/',
        pref_dir  = '/media/morio/Data_Fast/dmdm_temporalExpectation/preferences/',
    )

    HPC_PATHS = dict(
        npx_dir  = '/ceph/mrsic_flogel/public/projects'
                   '/MoHa_20260212_dmdmTemporalExpectation/data/npx/',
        ceph_dir = '/ceph/mrsic_flogel/public/projects'
                   '/MoHa_20260212_dmdmTemporalExpectation/data/npx/npx_converted/',
        plots_dir = '/ceph/mrsic_flogel/public/projects'
                    '/MoHa_20260212_dmdmTemporalExpectation/hpc/plots/',
    )
    PATHS = LOCAL_PATHS if os.path.exists(LOCAL_PATHS['npx_dir']) else HPC_PATHS
elif platform=='darwin':
    LOCAL_PATHS = dict(
        npx_dir   = '/Volumes/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx/',
        ceph_dir  = '/Volumes/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted/',
        plots_dir = '~/Documents/dmdm/temporal_expectation/dmdm_npx/plots/',
    )

    PATHS = LOCAL_PATHS

# old keys for compatibility
PATHS['npx_dir_local'] = PATHS['npx_dir']
PATHS['npx_dir_ceph']  = PATHS['ceph_dir']

ANALYSIS_OPTIONS = dict(
    sp_bin_width     = 10 / 1000,     # s
    sp_smooth_width  = 50 / 1000,     # s; size of casual boxcar fitler
    resp_buffer      = 500 / 1000,    # s; extra time before each PSTH window for smoothing

    tf_outlier = 1.0,                 # std deviations away from mean to consider an outlier

    min_trial_dur   = 1.5,            # s; minimum trial duration
    rmv_time_around_move = 1.5,       # s; remove this time around aborts/licks
    rmv_time_around_bl = 1.0,         # s; remove this time from baseline.

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

    # Behavioural analysis
    change_tfs    = [1, 1.25, 1.35, 1.5, 2, 4],
    change_wins   = {'early': [3, 8], 'late': [10.5, 15.5]},
    ignore_trial_start  = 2,               # s; ignore FAs in first seconds of trial
    ignore_first_sessions = 2,            # ignore first N sessions after introducing blocks

    smooth_tf          = 5,               # samples; moving average for lick-triggered analyses
    frame_rate         = 60,
    tf_sample_step     = 3,               # subsample every 3 frames
    tf_sample_rate     = 20,              # Hz; = 60/3
    n_pre_lick_samples = 40,              # 2s at 20Hz

    hazard_bin_size = 0.5,                # s
    hazard_bin_step = 0.1,                # s

    tf_pulse_bin_centres = np.arange(-1, 1, .025),
    tf_pulse_bin_width   = 0.1,           # octaves
    tf_pulse_lick_win    = [0.2, 1.5],    # s; time after TF pulse to count as 'licked'

    sig_thresh = 0.05,
)

GLM_OPTIONS = dict(
    bin_width = 50 / 1000,  # s; matches TF pulse duration (3 frames at 60Hz)

    # predictor kernel windows (start, end) in seconds
    # positive lags = predictor precedes response, negative = follows
    kern_tf          = (0, 1.0),
    kern_trial_start = (0, 2.0),
    kern_change      = (0, 2.0),
    kern_lick_prep   = (-1.25, 0),
    kern_lick_exec   = (0, 0.5),
    kern_air_puff    = (-0.5, 0.25),
    kern_reward      = (-0.5, 0.4),
    kern_abort       = (-1.25, 0.25),
    kern_face_me     = (-0.05, 0.8),
    kern_running     = (-0.05, 0.8),
    # kern_pupil removed - contains NaNs across sessions

    n_phase_bins = 12,

    # unit classification thresholds
    min_r = 0,
    lesion_alpha = 0.05,

    # PETH-based classification (paper's criteria)
    tf_sd_threshold = 0.5,   # pulse threshold in s.d. of baseline log2(TF)
    peth_r_thresh   = 0.2,   # criterion 1: mean across-fold pearson r
    peth_alpha      = 0.01,  # criterion 2: residual-prediction t-test alpha

    # predictor groups to lesion together for unit classification
    lesion_groups = {
        'tf':            ['tf'],
        'lick_prep':     ['lick_prep'],
        'lick_exec':     ['lick_exec'],
        'time_in_trial': ['time_ramp'],
        'block':         ['block'],
    },

    # regularisation
    group_lasso_lambdas = [0, 1e-4, 1e-3, 1e-2],
    ridge_lambdas = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],

    # cross-validation
    n_folds = 10,

    # fitting
    max_iter = 10000,
    tol = 1e-6,
    
    cv_max_iter = 1000,  # coarser convergence for lambda selection
    cv_tol = 1e-4,
)
#%%
NETWORK_OPTIONS = dict(
    # architecture
    hidden_sizes = [0, 16, 64, 128],  # 0 = PoissonLinear (no hidden layer)

    # regularisation
    group_lasso_lambdas = [1e-4, 1e-3, 1e-2],

    # training
    lr          = 1e-2,
    batch_size  = 4096,
    max_epochs  = 2500,
    patience    = 100,
    val_frac    = 0.1,
    cv_max_epochs = 800,   # coarser convergence for CV lambda selection
    cv_patience   = 40,

    # CV
    n_outer_folds = 10,
    n_inner_folds = 3,
    n_jobs        = 10,  # parallel OUTER CV jobs (match --cpus-per-task)

    # lesion groups (shared with GLM)
    lesion_groups = GLM_OPTIONS['lesion_groups'],
    lesion_alpha  = GLM_OPTIONS['lesion_alpha'],
    n_perm_importance = 10,  # permutation repeats for predictor importance

    # interaction combos to test via permutation
    # 2-way: all pairs of tf, lick_prep, time_in_trial, block
    # 3-way: only block x tf x time_in_trial
    interaction_combos = [
        ('block', 'tf'),
        ('block', 'time_in_trial'),
        ('tf', 'time_in_trial'),
        ('block', 'tf', 'time_in_trial'),
    ],

    # neuron selection
    min_r       = 0,
    require_tf  = False,
)

HYBRID_OPTIONS = dict(
    interactions = [
        ('tf', 'block'),
        ('block', 'time_in_trial'),
        ('tf', 'time_in_trial'),
        ('tf', 'block', 'time_in_trial'),
    ],
    units_per_group = 2,

    # regularisation
    group_lasso_lambdas = [0, 1e-4, 1e-3, 1e-2],

    # training
    lr          = 1e-2,
    batch_size  = 4096,
    max_epochs  = 2000,
    patience    = 50,
    val_frac    = 0.1,

    # CV
    n_outer_folds = 10,

    # lesion groups (for skip connection group lasso)
    lesion_groups = GLM_OPTIONS['lesion_groups'],
)
#%%
LICK_PRED_OPS = dict(
    bin_width       = 50 / 1000,       # s; prediction bin width (matches pop_bin_width)
    tf_history_bins = 40,              # number of 50ms bins of TF history (2s)
    tf_subsample    = 3,               # subsample raw 60Hz TF by this factor

    # target
    lick_sigma_bins   = 5,             # gaussian kernel sigma in bins
    lick_extend_bins  = 5,             # extend trial 250ms past lick
    response_window   = 2.15,          # s; task response window after change onset
    max_change_tf     = 1.5,           # Hz; include change period for changes <= this

    # trial history
    max_time_since_reward = 300,       # s; cap for first trial before any reward

    # training
    hidden_sizes    = [8, 16, 32, 64, 128],
    lambdas         = [0, 1e-3, 1e-2],       # weight decay for linear model
    ortho_lambdas   = [1e-3, 1e-2, 1e-1, 1.0],  # orthogonality penalty for networks
    net_sweep       = 'grid',                 # sweep 'ridge', 'ortho', or 'grid' (both)
    lr              = 1e-4,
    batch_size      = 4096,
    max_epochs      = 1000,
    sweep_epoch_frac = 0.2,           # fraction of max_epochs for quick sweep
    patience        = 50,             # early stopping patience (epochs)
    val_frac        = 0.1,             # fraction of training trials for early stopping
)
#%%
DEMIXING_OPTIONS = dict(
    model_type = 'sae',               # 'sae' or 'lfads'
    latent_dim = 25,                  # number of latent dimensions
    rnn_dim    = 100,                 # RNN hidden size (CausalLFADS only)

    loss        = 'MSE',
    l1_weight   = 5,
    orth_weight = 0,
    lr          = 1e-4,
    optimizer   = 'Adam',
    epochs      = 200,
    batch_size  = 10,
    test_frac   = 0.2,
)
#%%
TUNING_CURVE_OPS = dict(
    tf_resp_win       = (0.1, 0.4),     # s; response window for tuning curves
    n_tf_bins         = 15,             # quantile bins for tuning curves (equal pulses)
    n_permutations    = 500,            # shuffles for significance tests
    plot_during_extraction = False,
    trial_split_time  = 5,
)
#%%
CODING_DIM_OPS = dict(
    sliding_window_ms = 100,            # ms; causal boxcar width for FR smoothing
    n_permutations    = 500,            # shuffles for per-animal significance tests
    n_perm_across     = 500,           # shuffles for across-animals tests
    n_perm_pooled     = 500,             # shuffles for pooled pseudo-population tests
    min_neurons       = 5,               # minimum neurons per session after filtering for tf-responsiveness/area etc
    min_tf_events_per_block    = 500,    # min fast/slow TF outliers per block per session
    min_lick_events_per_block  = 20,     # min lick events per block per session
    min_block_trials_per_block = 30,     # min trials per block per session for block dims
    plot_during_extraction = False,
    trial_split_time  = 5,              # s; only use TF pulses before this time

    # TF coding direction windows (s, relative to pulse onset)
    tf_coding_windows = [(0.1, 0.3), (0.1, 0.4), (0.25, .6) ],

    # premotor coding direction windows (s, relative to lick onset)
    motor_prelick_windows = [(-1.0, -0.6), (-0.5, -0.2), (-0.25, 0.0)],
    motor_baseline_window = (-1.5, -1.0),

    # block coding direction windows (s, relative to baseline onset)
    # negative = pre-baseline (ITI)
    block_coding_windows = [(-2.0, 0.0), (0.0, 2.0), (2.0, 5.0)],
)
#%%
MOVEDIM_OPTIONS = dict(
    full_lick_period = (-2, 2), # full period to extract activity around licks
    lick_period = (-.5, 1),   # period to predict for movement potent
    prelick_period = (-1, 0), # period to capture premotor activity (pca on this period)
    min_licks = 20,
    min_neurons = 10,
)
#%%
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

        tf_pref = {
            'fast': (204/255,  37/255,  41/255),  # Red
            'slow': ( 57/255, 106/255, 177/255),  # Blue
        },

        ch_tf_cmap = 'viridis',
    ),

    smooth_window_short = 250 / 1000,  # s; causal boxcar window for short trajectories (TF, blOn)
    smooth_window_long  = 500 / 1000,  # s; causal boxcar window for long trajectories (bl, lick)
)


