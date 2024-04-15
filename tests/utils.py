import copy

import torch
import glob
from torch.utils.data import DataLoader

from deepinv.physics import Inpainting, Blur
from deepinv.physics.blur import gaussian_blur
from deepinv.datasets import generate_dataset, HDF5Dataset
from torchvision import transforms

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


class CH5Dataset(HDF5Dataset):
    def __init__(self, *args, img_size = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tr = None
        if img_size is not None:
            self.tr = transforms.CenterCrop(img_size)

    def __getitem__(self, item):
        x, y = super().__getitem__(item)
        if self.tr is not None:
            x = self.tr(x)
            y = self.tr(y)
        return x, y


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
                data_bis = CH5Dataset(img_size=params_exp["shape"][1:2], path=find_file, train=False)
            case _:
                # create dataset if it does not exist
                data_bis = generate_dataset(
                    train_dataset=None, physics=physics, save_dir=save_dir, device=device, test_dataset=input_data
                )
        data = DataLoader(data_bis, shuffle=False)

    return data


def single_level_params(params_ml):
    params = params_ml.copy()
    params['n_levels'] = 1
    params['level'] = 1
    params['iters'] = params_ml['params_multilevel'][0]['iters'][-1]
    params.pop('params_multilevel')

    return params

def standard_multilevel_param(params, it_vec):
    levels = len(it_vec)
    ml_dict = {"iters": it_vec}
    params['params_multilevel'] = [ml_dict]
    params['level'] = levels
    params['n_levels'] = levels

    #lambda_vec = [lambda_def / 4 ** (levels - 1 - i) for i in range(0, levels)]
    #stepsize_vec = [step_coeff / (l0 * lip_g + 1.0) for l0 in lambda_vec]

    return params


from prettytable import PrettyTable
def count_parameters(model, pr=True, namenet=''):
    table = PrettyTable(["Modules", "Parameters"])

    total_params = 0
    total_params_dict = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
        total_params_dict[name] = parameter

    if pr == True:
        print(table)

    print(namenet + " has total Trainable Params: ", total_params)

    return total_params, total_params_dict

