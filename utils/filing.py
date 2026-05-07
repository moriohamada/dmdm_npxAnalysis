"""
useful functions for navigating directories, getting paths for relevant files etc.
"""
import os
from pathlib import Path
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

def get_session_dirs_by_animal(npx_dir: str):
    """group session directories by animal, returns {animal: [sess_dir, ...]}"""
    psth_paths = get_response_files(npx_dir)
    animals = {}
    for path in psth_paths:
        sess_dir = Path(path).parent
        animal = sess_dir.parent.name
        if animal not in animals:
            animals[animal] = []
        animals[animal].append(sess_dir)
    return animals


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

    return session_paths


def file_suffix(area=None, unit_filter=None):
    """build filename suffix from area and unit filter"""
    parts = [area if area and area != 'all' else 'all']
    if unit_filter:
        parts.append('-'.join(unit_filter))
    return '_'.join(parts)


IPYTHON_AUTO_NAMES = frozenset({'In', 'Out', 'exit', 'quit', 'get_ipython'})


def save_workspace(path, ns=None, exclude=()):
    """pickle data variables from a namespace (ns) dict (default: caller globals)

    skips modules, callables, dunder names, and IPython auto-injected
    history (In, Out). variables that fail to pickle are reported and
    skipped rather than aborting the whole save.
    """
    import dill
    import inspect
    import types

    if ns is None:
        ns = inspect.currentframe().f_back.f_globals
    exclude = set(exclude) | IPYTHON_AUTO_NAMES

    out = {}
    skipped = []
    for k, v in ns.items():
        if k.startswith('_') or k in exclude:
            continue
        if isinstance(v, types.ModuleType) or callable(v):
            continue
        try:
            dill.dumps(v)
        except Exception as e:
            skipped.append((k, type(v).__name__, str(e).splitlines()[0]))
            continue
        out[k] = v

    with open(path, 'wb') as f:
        dill.dump(out, f)

    print(f'saved {len(out)} variables to {path}')
    for name, kind, msg in skipped:
        print(f'  skipped {name} ({kind}): {msg}')
    return list(out.keys())


def load_workspace(path, ns=None):
    """unpickle a workspace dict and inject it into a namespace (default: caller globals)"""
    import dill
    import inspect

    if ns is None:
        ns = inspect.currentframe().f_back.f_globals

    with open(path, 'rb') as f:
        ws = dill.load(f)
    ns.update(ws)
    print(f'loaded {len(ws)} variables: {sorted(ws)}')
    return list(ws.keys())
