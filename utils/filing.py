"""
useful functions for navigating directories, getting paths for relevant files etc.
"""
import os
import pandas as pd


def load_fr_matrix(path) -> pd.DataFrame:
    """Load large FR_matrix parquet files"""
    import pyarrow.parquet as pq
    table = pq.read_table(str(path), thrift_string_size_limit=1000000000)
    return table.to_pandas()

def get_response_files(npx_dir: str):
    """
    Get list of paths to psths for all sessions
    """
    psth_paths = []

    subj_folders = [f.path for f in os.scandir(npx_dir) if f.is_dir()]
    for subj_folder in subj_folders:
        sess_folders = [f.path for f in os.scandir(subj_folder) if f.is_dir()]
        for sess_folder in sess_folders:

            if os.path.exists(os.path.join(sess_folder, 'psths.h5')):
                psth_paths.append(os.path.join(sess_folder, 'psths.h5'))

    return psth_paths

def get_session_files(npx_dir: str,
                      npx_only = True):
    """
    Get list of paths for all sessions/all recording sessions
    """
    session_paths = []

    subj_folders = [f.path for f in os.scandir(npx_dir) if f.is_dir()]
    for subj_folder in subj_folders:
        sess_folders = [f.path for f in os.scandir(subj_folder) if f.is_dir()]
        for sess_folder in sess_folders:
            file_parts   = sess_folder.split('/')
            session_name = file_parts[-1]

            if npx_only and not os.path.exists(os.path.join(sess_folder,
                                                            'FR_matrix.parquet')):
                # not a recording
                continue

            if os.path.exists(os.path.join(sess_folder, 'session.pkl')):
                session_paths.append(os.path.join(sess_folder, 'session.pkl'))
            else:
                print(f'Session file not found for {session_name}')

    return session_paths