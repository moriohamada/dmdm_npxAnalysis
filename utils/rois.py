import numpy as np

# group -> list of areas (groups can overlap)
AREA_GROUPS = {
    'early_visual':   ['SCs', 'LGd', 'VISp'],
    'higher_visual':  ['LP', 'VISl', 'VISpl', 'VISa', 'VISam', 'VISrl', 'VISpm',
                        'PPC', 'RSP'],
    'basal_ganglia':  ['CP', 'GPe', 'SNr', 'GPi', 'LS'],
    'frontal_cortex': ['FRP', 'MOs', 'ACA', 'PL', 'ILA', 'mPFC', 'ORB', 'MOp', 'AI'],
    # 'olfactory':      ['MOB', 'DP', 'TTd'],
    'thalamus':       ['LD', 'CL', 'MD', 'VM', 'PF', 'VAL', 'PO', 'VB', 'VPL',
                        'VPM', 'MG', 'RT', 'Eth'],
    'midbrain':       ['IC', 'MRN', 'APN', 'SCm', 'NPC'],
    'hippocampus':    ['CA1', 'CA3', 'DG', 'POST', 'PRE', 'ProS', 'SUB', 'ENT'],
    'cerebellum':     ['Lob4/5', 'SIM', 'CRUS1', 'CRUS2', 'CENT3', 'FN', 'IP',
                        'DN', 'DCN', 'FL', 'PFL'],
    # 'hypothalamus':   ['LHA'],
    # 'medulla':        ['GRN', 'MV', 'IRN', 'SPV', 'V'],

    # composite
    'frontal_motor':  ['FRP', 'MOs', 'ACA', 'PL', 'ILA', 'mPFC', 'ORB', 'MOp', 'AI',
                        'CP', 'GPe', 'SNr', 'GPi', 'MD', 'VAL'],
}

# all areas that appear in at least one group
ALL_AREAS = set().union(*AREA_GROUPS.values())


def in_any_area(areas: np.ndarray) -> np.ndarray:
    """Return boolean mask for units belonging to any known area."""
    return np.array([a in ALL_AREAS for a in areas])


def in_group(areas: np.ndarray, group: str) -> np.ndarray:
    """Return boolean mask for units belonging to a specific group."""
    members = set(AREA_GROUPS[group])
    return np.array([a in members for a in areas])
