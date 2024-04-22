import torch
from deepinv.models import DRUNet
from torchvision import transforms

# install deepinv using the command below
#   python -m pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
# uninstall with :
#   python -m pip uninstall deepinv
# update by uninstall + install + restart IDE

import deepinv
from deepinv.physics import GaussianNoise
from deepinv.utils.demo import load_dataset

from tests.drunet_scale import test_drunet_scale
from tests.test_lipschitz import measure_lipschitz
from multilevel.info_transfer import BlackmannHarris
from tests.rastrigin import eval_rastrigin, test_rastrigin
from tests.test_alg import RunAlgorithm
from tests.utils import physics_from_exp, data_from_user_input
from tests.utils import standard_multilevel_param, single_level_params
from utils.param_grad import tune_param
from utils.param_grid import tune_grid_all
from utils.paths import dataset_path


def test_settings(data_in, params_exp, device):
    noise_pow = params_exp["noise_pow"]
    problem = params_exp['problem']
    print("def_noise:", noise_pow)

    g = GaussianNoise(sigma=noise_pow)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    if problem == "inpainting":
        lambda_tv = 2.5 * noise_pow
        lambda_red = 0.2 * noise_pow
    else:  # blur problem
        lambda_tv = 0.6 * noise_pow
        lambda_red = 0.05 * noise_pow

    print("lambda_tv:", lambda_tv)
    print("lambda_red:", lambda_red)

    lip_g = 160.0  # DRUnet lipschitz
    g_param = 0.05  # sigma denoiser

    iters_fine = 200
    lc = 3
    iters_vec = [lc, lc, lc, iters_fine]
    if device == "cpu":
        iters_fine = 5
        iters_vec = [2, 2, iters_fine]

    params_algo = {
        'cit': BlackmannHarris(),
        'iml_max_iter': 8,
        'scale_coherent_grad': True
    }

    #                    RED
    # ____________________________________________
    p_red = params_algo.copy()
    p_red = standard_multilevel_param(p_red, it_vec=iters_vec)
    p_red['g_param'] = g_param
    p_red['lip_g'] = lip_g  # denoiser Lipschitz constant
    p_red['lambda'] = lambda_red
    p_red['step_coeff'] = 0.9  # no convex setting
    p_red['stepsize'] = p_red['step_coeff'] / (1.0 + lambda_red * lip_g)

    param_init = {'init_ml_x0': [80] * len(iters_vec)}
    ra = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init)
    ra.RED_GD(p_red.copy())
    return
    ra.RED_GD(single_level_params(p_red.copy()))

    ra = RunAlgorithm(data, physics, params_exp, device=device)
    ra.RED_GD(p_red.copy())
    ra.RED_GD(single_level_params(p_red.copy()))

    #                    DPIR
    # ____________________________________________
    ra = RunAlgorithm(data, physics, params_exp, device=device)
    ra.DPIR(single_level_params(p_red))

    #                    PGD
    # ____________________________________________
    p_tv = params_algo.copy()
    p_tv = standard_multilevel_param(p_tv, it_vec=iters_vec)
    p_tv['lambda'] = lambda_tv
    p_tv['lip_g'] = 1.0  # denoiser Lipschitz constant
    p_tv['prox_crit'] = 1e-6
    p_tv['prox_max_it'] = 1000
    p_tv['params_multilevel'][0]['gamma_moreau'] = [1.1] * len(iters_vec)  # smoothing parameter
    p_tv['params_multilevel'][0]['gamma_moreau'][-1] = 1.0  # fine smoothing parameter
    p_tv['step_coeff'] = 1.9  # convex setting
    p_tv['stepsize'] = p_tv['step_coeff'] / (1.0 + lambda_tv)

    # todo: attention!! TV et TV multilevel n'ont pas la même cost car la réalisation de bruit est différente!!
    ra = RunAlgorithm(data, physics, params_exp, device=device)
    ra.TV_PGD(p_tv)
    p_tv['prox_crit'] = 1e-6
    p_tv['prox_max_it'] = 1000
    p_tv = single_level_params(p_tv.copy())
    ra.TV_PGD(p_tv)


def main_test(problem, test_dataset=True, tune=False):
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(device)

    dataset_name = 'set3c'
    original_data_dir = dataset_path()
    img_size = 256 if torch.cuda.is_available() else 64
    if tune is True:
        img_size = 32

    max_lv = 2
    val_transform = transforms.Compose(
        [transforms.CenterCrop(img_size), transforms.ToTensor()]
    )

    dataset = load_dataset(dataset_name, original_data_dir, transform=val_transform)

    # inpainting: proportion of pixels to keep
    if problem == 'inpainting':
        params_exp = {'problem': problem, 'set_name': dataset_name, problem: 0.5, 'noise_pow': 0.1, 'shape': (3, img_size, img_size)}
    elif problem == 'blur':
        params_exp = {'problem': problem, 'set_name': dataset_name, problem + '_pow': 2, 'noise_pow': 0.1, 'shape': (3, img_size, img_size)}
    else:
        raise NotImplementedError()

    if tune is True:
        #tune_param(dataset, params_exp, device, max_lv)
        tune_grid_all(dataset, params_exp, device, max_lv)
        return

    if test_dataset is True:
        test_settings(dataset, params_exp, device=device)
    else:
        id_img = 0
        for t in dataset:
            id_img += 1
            name_id = dataset_name + "_" + str(id_img)
            params_exp['img_name'] = name_id
            test_settings(t[0].unsqueeze(0).to(device), params_exp, device=device)
            #break


if __name__ == "__main__":
    # test_rastrigin()
    #main_test('inpainting', tune=True)
    #main_test('inpainting', test_dataset=False)
    #main_test('blur')
    test_drunet_scale()

    #device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    #denoiser = DRUNet(device=device)
    ##sigma_vec = [0.02, 0.1]
    #sigma_vec = [0.02 + n * 0.001 for n in range(0, 200)]
    #measure_lipschitz(denoiser, sigma_vec=sigma_vec, device=device)
