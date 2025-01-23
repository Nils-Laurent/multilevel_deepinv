from os.path import join, dirname
import sys
from pathlib import Path

from multilevel_utils.custom_poisson_noise import CPoissonLikelihood


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
    return CKPT_DIR

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
    if 'img_name' in p_exp.keys():
        exp = f"{p_exp['img_name']}_n{p_exp['noise_pow']}"
    else:
        exp = f"{p_exp['set_name']}_n{p_exp['noise_pow']}"

    problem = p_exp['problem']
    from utils.parameters import ConfParam

    if problem == 'inpainting':
        ir = ConfParam().inpainting_ratio
        exp += f"_{problem}{ir}"
    else:
        exp += f"_{problem}"

    if ConfParam().s1coherent_algorithm:
        coh = "ct_"
    else:
        coh = "cf_"
    if isinstance(ConfParam().data_fidelity(), CPoissonLikelihood):
        exp = coh + "pl_" + exp
    else:
        exp = coh + exp
    if ConfParam().use_equivariance is False:
        exp = "neq_" + exp

    from utils.parameters_global import FixedParams
    exp = FixedParams().get_str() + exp

    # identifies algorithm for resolution
    f_prefix = exp + f"_{alg_name}"

    if 'level' in params.keys() and params['level'][0] > 1:
        f_prefix += f"_{params['level'][0]}L{params['cit'][0]}"

    return f_prefix, exp
