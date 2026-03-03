
PATHS = dict(
    npx_dir_ceph = '/mnt/ceph/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx',
    npx_dir_local = '/media/morio/Data_Fast/dmdm_temporalExpectation/npx/',
)

ANALYSIS_OPTIONS = dict(
    sp_bin_width    = 10/1000,   # s
    sp_smooth_width = 50/1000,   # s; size of casual boxcar fitler
    rmv_time_around = 1.5,       # s; remove events with this time of other events

    tf_outlier      = 1.0,       # std deviations away from mean to consider an outlier

)