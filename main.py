import torch
from torchvision import transforms

# install deepinv using the command below
#   python -m pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
# update with :
#   python -m pip install -U git+https://github.com/deepinv/deepinv.git#egg=deepinv

import deepinv
from deepinv.physics import GaussianNoise
from deepinv.utils.demo import load_dataset

from multilevel.info_transfer import BlackmannHarris
from multilevel.iterator import MultiLevelParams
from tests.rastrigin import eval_rastrigin, test_rastrigin
from tests.test_alg import RunAlgorithm
from tests.utils import physics_from_exp, data_from_user_input
from tests.utils import standard_multilevel_param, single_level_params
from utils.param_grad import tune_param
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
        lambda_tv = 2.5 * noise_pow
        lambda_red = 0.05 * noise_pow

    print("lambda_tv:", lambda_tv)
    print("lambda_red:", lambda_red)

    lip_d = 160.0  # DRUnet
    g_param = 0.05  # 0.05

    iters_fine = 80
    iters_vec = [5, 5, 5, iters_fine]
    if device == "cpu":
        iters_vec = [5, 5, iters_fine]

    levels = len(iters_vec)

    params_algo = {
        'cit': BlackmannHarris(),
        'level': levels,
        'iml_max_iter': 8,
    }

    #                    RED
    # ____________________________________________
    p_red = standard_multilevel_param(params_algo, lambda_red, step_coeff=0.9, lip_g=lip_d, it_vec=iters_vec)
    p_red['g_param'] = g_param
    p_red['scale_coherent_grad'] = True

    param_init = {
        'init_ml_x0': [80] * levels,
        'lambda': lambda_red,
        'step_coeff': 0.9,
        'lip_g': lip_d,
    }
    ra = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init)
    ra.RED_GD(p_red)
    ra.RED_GD(single_level_params(p_red))

    #                    DPIR
    # ____________________________________________
    ra = RunAlgorithm(data, physics, params_exp, device=device)
    ra.DPIR(single_level_params(p_red))

    #                    PGD
    # ____________________________________________
    p_moreau = standard_multilevel_param(params_algo, lambda_tv, step_coeff=1.9, lip_g=1.0, it_vec=iters_vec)
    p_moreau['prox_crit'] = 1e-6
    p_moreau['prox_max_it'] = 1000
    p_moreau['params_multilevel'].params['gamma_moreau'] = [1.1] * levels  # smoothing parameter
    p_moreau['params_multilevel'].params['gamma_moreau'][-1] = 1.0  # fine smoothing parameter
    p_moreau['scale_coherent_grad'] = True

    ra = RunAlgorithm(data, physics, params_exp, device=device)
    ra.TV_PGD(p_moreau)
    ra.TV_PGD(single_level_params(p_moreau))


def main_test():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(device)

    dataset_name = 'set3c'
    original_data_dir = dataset_path()
    img_size = 256 if torch.cuda.is_available() else 64
    val_transform = transforms.Compose(
        [transforms.CenterCrop(img_size), transforms.ToTensor()]
    )

    dataset = load_dataset(dataset_name, original_data_dir, transform=val_transform)

    # inpainting: proportion of pixels to keep
    #problem = 'inpainting'
    #params_exp = {'problem': problem, 'set_name': dataset_name, problem: 0.8, 'noise_pow': 0.1, 'shape': (3, img_size, img_size)}
    problem = 'blur'
    params_exp = {'problem': problem, 'set_name': dataset_name, problem + '_pow': 2, 'noise_pow': 0.1, 'shape': (3, img_size, img_size)}

    bool_dataset = False
    # bool_dataset = True

    if bool_dataset is True:
        tune_param(dataset, params_exp, device)
        # test_settings(dataset, params_exp, device=device)
    else:
        id_img = 0
        for t in dataset:
            id_img += 1
            name_id = dataset_name + "_" + str(id_img)
            params_exp['img_name'] = name_id
            test_settings(t[0].unsqueeze(0).to(device), params_exp, device=device)
            break


if __name__ == "__main__":
    # test_rastrigin()
    main_test()
