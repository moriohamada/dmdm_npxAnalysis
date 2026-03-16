import numpy as np

# group -> list of areas (groups can overlap)
AREA_GROUPS = {
    'early_visual':   ['SCsl', 'SCdl', 'LGd', 'VISp'],
    'higher_visual':  ['LP', 'VISl', 'VISpl', 'VISa', 'VISam', 'VISrl', 'VISpm',
                        'VISpor', 'RSP', 'RSPd', 'RSPv'],
    'basal_ganglia':  ['CP', 'STR', 'ACB', 'GPe', 'GPi', 'SNr', 'SNc', 'LS'],
    'frontal_cortex': ['FRP', 'MOs', 'ACA', 'PL', 'ILA', 'ORB', 'MOp', 'AI'],
    # 'thalamus':       ['TH', 'LD', 'CL', 'MD', 'VM', 'PF', 'VAL', 'PO',
    #                     'VPL', 'VPM', 'MGd', 'MGv', 'RT', 'Eth',
    #                     'AV', 'SMT', 'PCN', 'SPF', 'PoT', 'PIL', 'ZI'],
    # 'midbrain':       ['IC', 'MRN', 'APN', 'SCiml', 'NPC',
    #                     'PAG', 'RN', 'VTA', 'NOT'],
    'hippocampus':    ['HPF', 'CA1', 'CA2', 'CA3', 'DG',
                        'POST', 'PRE', 'ProS', 'SUB',
                        'ENTl', 'ENTm', 'PAR'],
    'cerebellum':     ['CUL4 5', 'SIM', 'ANcr1', 'ANcr2', 'CENT',
                        'FN', 'IP', 'DN', 'FL', 'PFL',
                        'DEC', 'NOD', 'LING', 'CUN', 'COPY', 'CB'],
    #
    # 'frontal_motor':  ['FRP', 'MOs', 'ACA', 'PL', 'ILA', 'ORB', 'MOp', 'AI',
    #                     'CP', 'ACB', 'GPe', 'GPi', 'SNr', 'SNc', 'MD', 'VAL'],
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
