from pathlib import Path
from os.path import join
import utils.paths as paths
import torch

import numpy as np

def grid_search_npy_filename(suffix):
    out_dir = paths.get_out_dir()
    file_name = Path(out_dir) / ('tune_info' + '_' + suffix + '.npy')
    return file_name

def save_grid_tune_info(data, suffix):
    out_f = grid_search_npy_filename(suffix)
    np.save(out_f, data)
    return out_f


def load_variables_from_npy(in_f):
    return np.load(in_f, allow_pickle=True).item()