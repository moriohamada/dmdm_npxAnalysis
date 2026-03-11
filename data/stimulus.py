"""
Functions for getting stimulus types/times
"""
from config import ANALYSIS_OPTIONS
from data.session import Session
import numpy as np
import pandas as pd

def get_trials_from_block_start(session: Session) -> Session:
    """
    Add number of trials since a block switch (or start of session)
    as column in session.trials
    """
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
    session.trials['tr_in_block'] = tr_from_block_switch
    return session

def get_tf_outliers(session: Session,
                    ops: dict = ANALYSIS_OPTIONS) -> Session:

    tf_pulses = []
    for tr, row in session.trials.iterrows():
        # first some basic info about the trial and lick
        block = row['hazardblock']
        tr_in_block = row['tr_in_block']
        tr_outcome = row['trialoutcome']
        if row['IsFA']:
            tr_lick = row['motion_onset'] \
                if not pd.isna(row['motion_onset']) \
                else (row['Baseline_ON_rise'] + row['rt_FA'])
        elif row['IsHit']:
            tr_lick = row['motion_onset']
        else:
            tr_lick = np.nan
        tr_abort = row['rt_abort']

        tf_seq = row['TF'][row['TF'].nonzero()]
        ch_t   = row['stimT']
        ch_fr  = round(ch_t * 60)
        bl_tf  = np.log2(tf_seq[:ch_fr:3])
        fr_t   = (row['frame_time']
                 [~np.isnan(row['frame_time'])][:ch_fr:3])

        if len(fr_t) != len(bl_tf):
            print(
                f'  Warning: skipping trial {tr} - fr_t/bl_tf length mismatch ({len(fr_t)} vs {len(bl_tf)})')
            continue

        outliers = np.where(np.abs(bl_tf) > ops['tf_outlier']*0.25)[0]

        if outliers.size == 0:
            continue

        tf = bl_tf[outliers]
        time = fr_t[outliers]
        time_to_lick = tr_lick - time
        time_to_abort = tr_abort - time
        time_in_tr = time - row['Baseline_ON_rise']

        tf_pulses.extend([{
            'tf': tf[i],
            'time': time[i],
            'tr_time': time_in_tr[i],
            'trial': tr,
            'block': block,
            'tr_in_block': tr_in_block,
            'tr_outcome': tr_outcome,
            'time_to_lick': time_to_lick[i],
            'time_to_abort': time_to_abort[i],
        } for i,_ in enumerate(outliers)])

    session.tf_pulses = pd.DataFrame(tf_pulses)

    return session

def get_baseline_onset_times(session:Session) -> Session:
    bl_onsets = (session.daq[
                 session.daq.event_type=='Baseline_ON']
                 .reset_index()
                 .drop(columns=['index', 'event_type'])
                 )
    block_id = session.trials['hazardblock']
    session.bl_onsets = pd.DataFrame()
    session.bl_onsets['time']  = bl_onsets['rise_t'].to_numpy()
    session.bl_onsets['block'] = block_id.to_list()
    session.bl_onsets['tr_dur'] = bl_onsets['duration'].to_numpy()
    session.bl_onsets['tr_in_block'] = session.trials['tr_in_block'].to_numpy()
    return session

def get_change_onset_times(session: Session) -> Session:
    ch_onsets = []
    for tr, row in session.trials.iterrows():
        if not row['IsHit'] and not row['IsMiss']:
             continue

        ch_onsets.append({
            'time': row['Change_ON_rise'],
            'tr_time': row['stimT'],
            'ch_tf': row['Stim2TF'],
            'trial': tr,
            'block': row['hazardblock'],
            'tr_in_block': row['tr_in_block'],
            'is_hit': row['IsHit'],
            'is_probe': row['IsProbe'],
        })
    session.ch_onsets = pd.DataFrame(ch_onsets)
    return session

def get_lick_onset_times(session: Session) -> Session:
    lick_onsets = []

    for tr, row in session.trials.iterrows():
        if not row['IsHit'] and not row['IsFA']:
            continue
        if np.isnan(row['motion_onset']):
            continue

        # get stimulus sequence leading up to lick
        window_fr = round(2 * 60 / 3)  # number of samples in 2s window at /3 subsampling
        lick_fr = round((row['motion_onset'] -
                        row['Baseline_ON_rise']) * 60)
        start_fr = max(0, lick_fr - round(2 * 60))

        tf_seq = row['TF'][row['TF'].nonzero()]
        lick_tf = np.log2(tf_seq[start_fr:lick_fr:3])

        # pad front with NaNs if window is shorter than 2s
        pad = window_fr - len(lick_tf)
        if pad > 0:
            lick_tf = np.concatenate([np.full(pad, np.nan), lick_tf])

        lick_onsets.append({
            'time': row['motion_onset'],
            'tr_time': row['motion_onset'] - row['Baseline_ON_rise'],
            'trial': tr,
            'block': row['hazardblock'],
            'tr_in_block': row['tr_in_block'],
            'is_hit': row['IsHit'],
            'is_FA': row['IsFA'],
            'is_probe': row['IsProbe'],
            'preceding_tf': lick_tf,
        })
    session.lick_times = pd.DataFrame(lick_onsets)
    return session