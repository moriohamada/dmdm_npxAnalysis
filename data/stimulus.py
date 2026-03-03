"""
Functions for getting stimulus types/times
"""
from config import ANALYSIS_OPTIONS
from data.session import Session
import numpy as np
import pandas as pd

def get_trials_from_block_start(session: Session) -> Session:
    block = session.trials['hazardblock']
    trans = [tr for tr, b in enumerate(block)
             if tr==0 or b!=block[int(tr-1)]]
    tr_from_block_switch = []
    count = 0
    for tr in range(len(block)):
        if tr in trans:
            count = 0
        else:
            count += 1
        tr_from_block_switch.append(count)
    session.trials['trInBlock'] = tr_from_block_switch
    return session


def get_tf_outliers(session: Session,
                    ops: dict = ANALYSIS_OPTIONS):

    tf_pulses = []
    for tr, row in session.trials.iterrows():
        # first some basic info about the trial and lick
        block = session.trials['hazardblock']
        tr_in_block = session.trials['trInBlock']
        tr_outcome = session.trials['trialoutcome']


        tf_seq = row['TF'][row['TF'].nonzero()]
        ch_t   = row['stimT']
        ch_fr  = round(ch_t * 60)
        bl_tf  = np.log2(tf_seq[:ch_fr:3])
        fr_t   = (row['frame_time']
                 [~np.isnan(row['frame_time'])][:ch_fr:3])
        assert len(fr_t)==len(bl_tf)

        outliers = np.where(np.abs(bl_tf) > ops['tf_outlier']*0.25)



def get_baseline_onset_times(session:Session,
                             ops: dict = ANALYSIS_OPTIONS):
    bl_onsets = (session.daq[
                 session.daq.event_type=='Baseline_ON']
                 .reset_index()
                 .drop(columns=['index', 'event_type'])
                 )
    block_id = session.trials['hazardblock']
    session.bl_onsets = pd.DataFrame()
    session.bl_onsets['time']  = bl_onsets['rise_t'].to_numpy()
    session.bl_onsets['block'] = block_id.to_list()
    session.bl_onsets['trDur'] = bl_onsets['duration'].to_numpy()
    session.bl_onsets['trInBlock'] = session.trials['trInBlock'].to_numpy()
    return session

def get_change_onset_times():
    raise NotImplementedError

def get_lick_onset_times():
    raise NotImplementedError
