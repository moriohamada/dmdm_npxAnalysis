import pandas as pd
import parquet
import numpy as np
from data.load_npx import extract_session_data
from config import *

#%% get all event times and event-aligned responses

extract_session_data(npx_dir_ceph = PATHS['npx_dir_ceph'],
                     npx_dir_local = PATHS['npx_dir_local'],
                     ops = ANALYSIS_OPTIONS)

#%% plot single unit psths



#%% PCA

#%% SAE
