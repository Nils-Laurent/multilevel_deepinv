from os.path import join, dirname
import sys
from pathlib import Path


def dataset_path():
    BASE_DIR = Path(".")  # emplacement du script qui contient __main__
    ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
    return ORIGINAL_DATA_DIR

def measurements_path():
    BASE_DIR = Path(".")  # emplacement du script qui contient __main__
    DATA_DIR = BASE_DIR / "measurements"
    return DATA_DIR

def checkpoint_path():
    BASE_DIR = Path(".")  # emplacement du script qui contient __main__
    CKPT_DIR = BASE_DIR / "ckpts"

def get_out_dir():
    r"""
    :return: path to figures and .mat directory
    """
    base_path = dirname(sys.argv[0])
    out_dir = join(base_path, "out")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir


def gen_fname(params, p_exp, alg_name):

    # identifies experiment
    exp = f"{p_exp['img_name']}_n{p_exp['noise_pow']}"

    problem = p_exp['problem']
    exp += f"_{problem}"

    # identifies algorithm for resolution
    f_prefix = exp + f"_{alg_name}"
    if 'level' in params.keys():
        f_prefix += f"_{params['level']}L{params['cit']}"

    # "_d{params['d']}"

    return f_prefix, exp
