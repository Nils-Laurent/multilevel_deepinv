import copy

import torch
import glob
from torch.utils.data import DataLoader

from deepinv.physics import Inpainting, Blur
from deepinv.physics.blur import gaussian_blur
from deepinv.datasets import generate_dataset, HDF5Dataset

from multilevel.iterator import MultiLevelIteration, MultiLevelParams
from utils.paths import measurements_path


def physics_from_exp(params_exp, noise_model, device):
    noise_pow = params_exp['noise_pow']
    problem = params_exp['problem']

    match problem:
        case 'inpainting':
            def_mask = params_exp[problem]
            print("def_mask:", def_mask)
            problem_full = problem + "_" + str(def_mask) + "_" + str(noise_pow)
            physics = Inpainting(params_exp['shape'], mask=def_mask, noise_model=noise_model, device=device)
        case 'blur':
            power = params_exp[problem + '_pow']
            print("def_blur_pow:", power)
            problem_full = problem + "_" + str(power) + "_" + str(noise_pow)
            physics = Blur(gaussian_blur(sigma=(power, power), angle=0), noise_model=noise_model, device=device)
        case _:
            raise NotImplementedError("Problem " + problem + " not supported")

    return physics, problem_full


def data_from_user_input(input_data, physics, params_exp, problem_name, device):
    if isinstance(input_data, torch.Tensor):
        data = input_data
    else:
        save_dir = measurements_path().joinpath(params_exp['set_name'], problem_name)
        f_prefix = str(save_dir.joinpath('**', '*.'))
        find = ""
        find_file = ""
        for filename in glob.iglob(f_prefix + 'h5', recursive=True):
            print(filename)
            find = "h5"
            find_file = filename

        match find:
            case 'h5':
                data_bis = HDF5Dataset(path=find_file, train=False)
            case _:
                # create dataset if it does not exist
                data_bis = generate_dataset(
                    train_dataset=None, physics=physics, save_dir=save_dir, device=device, test_dataset=input_data
                )
        data = DataLoader(data_bis, shuffle=False)

    return data


def single_level_params(param_multilevel):
    p_sl = MultiLevelIteration.get_level_params(param_multilevel)
    out = {}
    for k_ in p_sl.keys():
        if k_ == 'params_multilevel':
            continue
        if k_ == 'level':
            continue
        out[k_] = p_sl[k_]
    return out


def standard_multilevel_param(params_in, lambda_def, step_coeff, lip_g, it_vec):
    params = copy.deepcopy(params_in)

    levels = len(it_vec)
    params['params_multilevel'] = MultiLevelParams({"iters": it_vec})
    params['level'] = levels

    lambda_vec = [lambda_def / 4 ** i for i in range(0, levels)]
    lambda_vec.reverse()
    params['params_multilevel'].params['lambda'] = lambda_vec

    stepsize_vec = [step_coeff / (l0 * lip_g + 1.0) for l0 in lambda_vec]
    params['params_multilevel'].params['stepsize'] = stepsize_vec

    params['params_multilevel'].params['verbose'] = [False] * levels
    params['params_multilevel'].params['verbose'][levels - 1] = True

    params['scale_coherent_grad'] = True

    return params