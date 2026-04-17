from config import PATHS, ANALYSIS_OPTIONS
from utils.stats import l2_normalise
import numpy as np


#%% Import lick-aligned psths (single event)

#%% get lick-sensitive units;

#%% Single trial baseline trajectories (along motor dim)?


def _load_baseline_activity():
    # load in single trial baseline trajs - need to go from raw FR
    pass

def _get_dim_by_session(dim_dict: dict, dim_type: str) -> dict:
    """
    re-organise dim by session, including unit ids
    """
    dim_by_sess = dict()
    for animal in dim_dict.keys():
        dim_by_sess[animal] = dict()
        sessions = dim_dict[animal]['included_sessions']
        dims = dim_dict[animal]['dimensions'][dim_type]
        has_block_split = 'early' in dims

        uids = {}
        start = 0
        for sid, count in enumerate(dim_dict[animal]['n_neurons_per_session']):
            uids[str(sid)] = list(range(start, start + count))
            start += count

        for sid, session in enumerate(sessions):
            dim_by_sess[animal][session] = dict()
            sess_ids = uids[str(sid)]

            dim_by_sess[animal][session]['unit_ids'] = (
                [uid for _, uid in [dim_dict[animal]['unit_ids'][i] for i in sess_ids]])

            dim_by_sess[animal][session]['ws'] = dict()

            if has_block_split:
                dim_by_sess[animal][session]['n_events'] = dict()
                time_keys = dims['early'].keys()
                for blk in ['early', 'late']:
                    dim_by_sess[animal][session]['ws'][blk] = dict()
                    for tk in time_keys:
                        w_sess, _ = l2_normalise(dims[blk][tk][sess_ids])
                        dim_by_sess[animal][session]['ws'][blk][tk] = w_sess
                    sn_blk = dim_dict[animal]['sess_n'][blk]
                    if isinstance(sn_blk, dict):
                        dim_by_sess[animal][session]['n_events'][blk] = {
                            ev: n[sid] for ev, n in sn_blk.items()}
                    else:
                        dim_by_sess[animal][session]['n_events'][blk] = sn_blk[sid]
            else:
                for tk in dims.keys():
                    w_sess, _ = l2_normalise(dims[tk][sess_ids])
                    dim_by_sess[animal][session]['ws'][tk] = w_sess

    return dim_by_sess

def _load_dims(npx_dir: str = PATHS['npx_dir_local'],
               ops=ANALYSIS_OPTIONS,
               area: str|None = None,
               unit_filter: str|None = None,
               dim_type: str = 'dprime_cd'
               ):
    # lodd in movement-potent/null dimensions
    from coding_dims.analysis import load_dimension_results
    tf_cd  = load_dimension_results(dim_type='tf', npx_dir=npx_dir,
                                   area=area, unit_filter=unit_filter)
    mov_cd = load_dimension_results(dim_type='motor', npx_dir=npx_dir,
                                    area=area, unit_filter=unit_filter)
    blk_cd = load_dimension_results(dim_type='block', npx_dir=npx_dir,
                                    area=area, unit_filter=unit_filter)

    dims = dict(tf = _get_dim_by_session(tf_cd, dim_type),
                motor = _get_dim_by_session(mov_cd, dim_type),
                block = _get_dim_by_session(blk_cd, dim_type))
    return dims


def _project_activity():
    # project psth/activity or FR onto dimension
    pass

def visualize_single_trial_activity():
    """
    visualize individual licks, baseline trajectories, and tf pulse responses? along
    dimension(s) of interest (sensory, motor related, block-related) for single session
    """
    pass

#%%

def visualize_prelick_activity_vs_evidence():
    """
    visualize relationship between dimension and 'amount of evidence' in defined window
    preceding licks
    """
    pass


#%%
from utils.rois import AREA_GROUPS
from utils.filing import get_session_dirs_by_animal, load_fr_matrix
from data.session import Session
import pandas as pd
import gc

areas = list(AREA_GROUPS.keys())
unit_filter = None

dims = _load_dims(npx_dir=PATHS['npx_dir_local'],
                  ops=ANALYSIS_OPTIONS,
                  area='mos',
                  unit_filter=None,
                  )

# load FR matrices and project onto dims
animal_sessions = get_session_dirs_by_animal(npx_dir=PATHS['npx_dir_local'])

for animal, sess_dirs in animal_sessions.items():
    for sess_dir in sess_dirs:
        print(f'{animal}/{sess_dir.name}')

        # if not all(animal in dims[dt] and sess_dir.name in dims[dt][animal]
        #              for dt in dims):
        #       continue
        missing_in = [dt for dt in dims
                      if animal not in dims[dt] or sess_dir.name not in dims[dt][animal]]
        if missing_in:
            print(f'  not in dims: {missing_in}')
            continue
        sess = Session.load(str(sess_dir / 'session.pkl'))
        fr = pd.read_parquet(sess_dir / 'FR_matrix_ds.parquet')

        session = sess_dir.name

        unit_ids = dims['tf'][animal][sess_dir.name]['unit_ids']
        fr_sub = fr.loc[unit_ids]
        del fr
        gc.collect()

        # make sure fr nN matches dim length
        n_dim = len(next(iter(dims['tf'][animal][sess_dir.name]['ws']['early'].values())))
        if fr_sub.shape[0] != n_dim:
            print(f'  shape mismatch: fr={fr_sub.shape[0]}, dim={n_dim}')
            continue

        print('    all looks ok!')