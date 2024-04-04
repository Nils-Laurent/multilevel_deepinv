import copy
import glob

import torch
from deepinv.datasets import generate_dataset, HDF5Dataset
from deepinv.physics.blur import gaussian_blur
from torch.utils.data import DataLoader
from torchvision import transforms

# install deepinv using the command below
#   python -m pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
# update with :
#   python -m pip install -U git+https://github.com/deepinv/deepinv.git#egg=deepinv

import deepinv
from deepinv.physics import GaussianNoise, Inpainting, Blur
from deepinv.utils.demo import load_dataset

from multilevel.info_transfer import BlackmannHarris
from multilevel.iterator import MultiLevelIteration, MultiLevelParams
from tests.test_alg import RunAlgorithm
from utils.paths import dataset_path, measurements_path


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


def standard_multilevel_param(params, lambda_def, step_coeff, lip_g):
    levels = params['level']
    lambda_vec = [lambda_def / 4 ** i for i in range(0, levels)]
    lambda_vec.reverse()

    params['params_multilevel'].params['lambda'] = lambda_vec

    stepsize_vec = [step_coeff / (l0 * lip_g + 1.0) for l0 in lambda_vec]
    params['params_multilevel'].params['stepsize'] = stepsize_vec

    params['params_multilevel'].params['verbose'] = [False] * levels
    params['params_multilevel'].params['verbose'][levels - 1] = True

    params['scale_coherent_grad'] = True


def test_settings(data_in, params_exp, device):

    problem = params_exp['problem']
    print("def_mask:", params_exp[problem])
    print("def_noise:", params_exp["noise_pow"])

    g = GaussianNoise(sigma=params_exp["noise_pow"])
    match problem:
        case 'inpainting':
            problem_full = problem + "_" + str(params_exp[problem]) + "_" + str(params_exp["noise_pow"])
            physics = Inpainting(params_exp['shape'], mask=params_exp[problem], noise_model=g, device=device)
        case 'blur':
            power = params_exp[problem + '_pow']
            problem_full = problem + "_" + str(power) + "_" + str(params_exp["noise_pow"])
            physics = Blur(gaussian_blur(sigma=(power, power), angle=0), device=device)
        case _:
            raise NotImplementedError("Problem " + problem + " not supported")

    if isinstance(data_in, torch.Tensor):
        data = data_in
    else:
        save_dir = measurements_path().joinpath(params_exp['set_name'], problem_full)
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
                    train_dataset=None, physics=physics, save_dir=save_dir, device=device, test_dataset=data_in
                )
        data = DataLoader(data_bis, shuffle=False)

    lambda_tv = 2.5 * params_exp["noise_pow"]
    lambda_red = 0.2 * params_exp["noise_pow"]

    print("lambda_tv:", lambda_tv)
    print("lambda_red:", lambda_red)

    lip_d = 160.0  # DRUnet
    g_param = 0.05  # 0.05

    iters_fine = 400
    iters_vec = [5, 5, 5, iters_fine]
    if device == "cpu":
        iters_vec = [5, 5, iters_fine]

    levels = len(iters_vec)
    p_multilevel = MultiLevelParams({"iters": iters_vec})

    params_algo = {
        'cit': BlackmannHarris(),
        'level': levels,
        'params_multilevel': p_multilevel,
        'iml_max_iter': 6,
    }

    #                    RED
    # ____________________________________________
    p_red = copy.deepcopy(params_algo)
    standard_multilevel_param(p_red, lambda_red, step_coeff=0.9, lip_g=lip_d)
    p_red['g_param'] = g_param
    p_red['scale_coherent_grad'] = True

    param_init = copy.deepcopy(p_red)
    param_init['init_ml_x0'] = [80] * levels

    ra = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init)
    ra.RED_GD(p_red)
    ra.RED_GD(single_level_params(p_red))

    #                    DPIR
    # ____________________________________________
    ra = RunAlgorithm(data, physics, params_exp, device=device)
    ra.DPIR(single_level_params(p_red))

    #                    PGD
    # ____________________________________________
    p_moreau = copy.deepcopy(params_algo)
    p_moreau['prox_crit'] = 1e-6
    p_moreau['prox_max_it'] = 1000
    p_moreau['params_multilevel'].params['gamma_moreau'] = [1.1] * levels  # smoothing parameter
    p_moreau['params_multilevel'].params['gamma_moreau'][-1] = 1.0  # fine smoothing parameter
    p_moreau['scale_coherent_grad'] = True
    standard_multilevel_param(p_moreau, lambda_tv, step_coeff=1.9, lip_g=1.0)

    ra = RunAlgorithm(data, physics, params_exp, device=device)
    ra.TV_PGD(p_moreau)
    ra.TV_PGD(single_level_params(p_moreau))


def main_test():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    dataset_name = 'set3c'
    original_data_dir = dataset_path()
    img_size = 256 if torch.cuda.is_available() else 64
    val_transform = transforms.Compose(
        [transforms.CenterCrop(img_size), transforms.ToTensor()]
    )

    dataset = load_dataset(dataset_name, original_data_dir, transform=val_transform)

    # inpainting: proportion of pixels to keep
    problem = 'inpainting'
    params_exp = {'problem': problem, 'set_name': dataset_name, problem: 0.8, 'noise_pow': 0.1, 'shape': (3, img_size, img_size)}
    test_settings(dataset, params_exp, device=device)
    return

    id_img = 0
    for t in dataset:
        id_img += 1
        if id_img != 2:
            continue
        name_id = dataset_name + "_" + str(id_img)
        params_exp['img_name'] = name_id
        test_settings(t[0].unsqueeze(0).to(device), params_exp, device=device)


if __name__ == "__main__":
    main_test()
