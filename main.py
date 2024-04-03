import copy
import torch
from torchvision import transforms
from pathlib import Path

# install deepinv using the command below
# pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv

import deepinv
from deepinv.physics import GaussianNoise, Inpainting
from deepinv.physics.blur import gaussian_blur, Blur
from deepinv.utils.demo import load_dataset

from optim.info_transfer import BlackmannHarris
from optim.optim_iterators.multi_level import MultiLevelIteration, MultiLevelParams
from tests.test_alg import RunAlgorithm


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


def test_settings(data, params_exp, device):
    if device is None:
        device = data.device

    print("def_mask:", params_exp["mask"])
    print("def_noise:", params_exp["noise_pow"])

    g = GaussianNoise(sigma=params_exp["noise_pow"])
    # physics = Blur(gaussian_blur(sigma=(2, 2), angle=0), device=device)
    physics = Inpainting(params_exp['shape'], mask=params_exp["mask"], noise_model=g, device=device)

    lambda_tv = 2.5 * params_exp["noise_pow"]
    lambda_red = 0.2 * params_exp["noise_pow"]

    print("lambda_tv:", lambda_tv)
    print("lambda_red:", lambda_red)

    lip_d = 160.0  # DRUnet
    g_param = 0.05  # 0.05

    iters_fine = 400
    iters_vec = [5, 5, 5, iters_fine]
    levels = len(iters_vec)
    p_multilevel = MultiLevelParams({"iters": iters_vec})

    params_algo = {
        'cit': BlackmannHarris(),
        'level': levels,
        'params_multilevel': p_multilevel,
    }

    #                    RED
    # ____________________________________________
    p_red = copy.deepcopy(params_algo)
    standard_multilevel_param(p_red, lambda_red, step_coeff=0.9, lip_g=lip_d)
    p_red['iml_max_iter'] = 5
    p_red['g_param'] = g_param
    p_red['scale_coherent_grad'] = True

    param_init = copy.deepcopy(p_red)
    param_init['init_ml_x0'] = [40] * levels
    # param_init['x0'] = data
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
    p_moreau['iml_max_iter'] = 6
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
    ORIGINAL_DATA_DIR = Path(".") / "datasets"
    # img_size = 256 if torch.cuda.is_available() else 32
    img_size = 256
    val_transform = transforms.Compose(
        [transforms.CenterCrop(img_size), transforms.ToTensor()]
    )

    dataset = load_dataset(dataset_name, ORIGINAL_DATA_DIR, transform=val_transform)

    params_exp = {'mask': 0.8, 'noise_pow': 0.1, 'shape': (3, img_size, img_size)}
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
