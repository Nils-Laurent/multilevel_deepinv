import cProfile
import torch.profiler as profiler
import os
import sys
from itertools import product

from torch.utils.data import Subset

from tests.utils_frequency import plot_spectr_ratio
from utils.get_hyper_param import inpainting_hyper_param, blur_hyper_param, tomography_hyper_param

if "/.fork" in sys.prefix:
    sys.path.append('/projects/UDIP/nils_src/deepinv')

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

from tests.test_lipschitz import measure_lipschitz
from multilevel.info_transfer import BlackmannHarris
from tests.test_alg import RunAlgorithm
from tests.utils import physics_from_exp, data_from_user_input
from tests.utils import standard_multilevel_param, single_level_params
from utils.npy_utils import save_grid_tune_info, load_variables_from_npy, grid_search_npy_filename
from utils.param_grid import tune_grid_all, tune_scatter_2d, tune_plot_1d
from utils.paths import dataset_path, get_out_dir


def test_settings(data_in, params_exp, device, benchmark=False):
    noise_pow = params_exp["noise_pow"]
    problem = params_exp['problem']

    if type(data_in) == torch.Tensor:
        print("Single image : using torch.manual_seed")
        params_exp["manual_seed"] = True
        #torch.manual_seed(0)

    print("def_noise:", noise_pow)

    tensor_np = torch.tensor(noise_pow).to(device)
    g = GaussianNoise(sigma=tensor_np)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    if problem == 'inpainting':
        hp_red, hp_tv = inpainting_hyper_param(noise_pow)
    elif problem == 'blur':
        hp_red, hp_tv = blur_hyper_param(noise_pow)
    elif problem == 'tomography':
        hp_red, hp_tv = tomography_hyper_param(noise_pow)
    else:
        raise NotImplementedError("not implem")

    lambda_tv = hp_tv['lambda']
    lambda_red = hp_red['lambda']
    g_param = hp_red['g_param']

    print("lambda_tv:", lambda_tv)
    print("lambda_red:", lambda_red)

    lip_g = 160.0  # DRUnet lipschitz

    iters_fine = 200
    #iters_fine = 200 should be 500
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
    ra = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init, return_timer=benchmark)

    #out_dir = get_out_dir()
    #out_f = os.path.join(out_dir, 'cprofile_data')
    #prof = torch.profiler.profile(
    #    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #    on_trace_ready=torch.profiler.tensorboard_trace_handler(out_f),
    #    with_modules=True)
    #prof.start()
    #for step in range(1 + 1 + 3):
    #    prof.step()
    #    ra.RED_GD(p_red.copy())
    #prof.stop()
    #cProfile.runctx(
    #    statement='ra.RED_GD(param)',
    #    globals={'param' : p_red.copy(), 'ra': ra},
    #    locals={},
    #    filename=out_f
    #)

    #ra.RED_GD(single_level_params(p_red.copy()))

    #ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    #ra.RED_GD(p_red.copy())
    #ra.RED_GD(single_level_params(p_red.copy()))

    #                    DPIR
    # ____________________________________________
    #ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    #ra.DPIR(single_level_params(p_red))

    #                    PGD
    # ____________________________________________

    # adjust parameters : proximal gradient descent
    params_algo['iml_max_iter'] = 1
    params_algo['scale_coherent_grad'] = True

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

    # /!\ sans torch.manual_seed, TV et TV multilevel n'ont pas la même cost car la réalisation de bruit est différente
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    ra.TV_PGD(p_tv)
    p_tv['prox_crit'] = 1e-6
    p_tv['prox_max_it'] = 1000
    p_tv = single_level_params(p_tv.copy())
    ra.TV_PGD(p_tv)


def main_test(
        problem,
        test_dataset=True,
        tune=False,
        benchmark=False,
        noise_pow=0.1,
        dataset_name='set3c',
        nb_subset=None):
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(device)

    original_data_dir = dataset_path()
    img_size = 256 if torch.cuda.is_available() else 64

    val_transform = transforms.Compose(
        [transforms.CenterCrop(img_size), transforms.ToTensor()]
    )

    dataset = load_dataset(dataset_name, original_data_dir, transform=val_transform)
    if nb_subset is not None:
        dataset = Subset(dataset, range(0, nb_subset))

    # inpainting: proportion of pixels to keep
    params_exp = {'problem': problem, 'set_name': dataset_name, 'shape': (3, img_size, img_size)}
    if problem == 'inpainting':
        params_exp[problem] = 0.5
        params_exp['noise_pow'] = noise_pow
    elif problem == 'tomography':
        params_exp[problem] = 0.6
        params_exp['noise_pow'] = noise_pow
    elif problem == 'blur':
        params_exp[problem + '_pow'] = 2.0
        params_exp['noise_pow'] = noise_pow
    else:
        raise NotImplementedError()

    if tune is True:
        return tune_grid_all(dataset, params_exp, device)

    if test_dataset is True:
        test_settings(dataset, params_exp, device=device, benchmark=benchmark)
    else:
        id_img = 0
        for t in dataset:
            id_img += 1
            name_id = dataset_name + "_" + str(id_img)
            params_exp['img_name'] = name_id

            img = t[0].unsqueeze(0).to(device)
            test_settings(img, params_exp, device=device, benchmark=benchmark)
            break


def main_tune(plot_and_exit=False):
    pb_list = ['inpainting', 'blur', 'tomography']
    noise_pow_vec = [0.05, 0.1, 0.2, 0.3]

    if plot_and_exit is True:
        main_tune_plot(pb_list, noise_pow_vec)
        return

    for pb, noise_pow in product(pb_list, noise_pow_vec):
        r_pb = main_test(pb, noise_pow=noise_pow, tune=True)
        file_pb = save_grid_tune_info(data=r_pb, suffix=pb + str(noise_pow))

    print("_____________________________")
    print("grid search finished")
    print("_____________________________")
    for pb, noise_pow in product(pb_list, noise_pow_vec):
        file_pb = grid_search_npy_filename(pb + str(noise_pow))
        data = load_variables_from_npy(file_pb)
        print(f"{pb}:")
        print(f"p_red* = {data['res_red']}")
        print(f"p_tv* = {data['res_tv']}")

def main_tune_plot(pb_list, noise_pow_vec):
    for pb, noise_pow in product(pb_list, noise_pow_vec):
        print("main_tune_plot", pb, noise_pow)
        file_pb = grid_search_npy_filename(suffix=pb + str(noise_pow))
        data = load_variables_from_npy(file_pb)

        data_red = data['data_red']
        keys_red = data['keys_red']
        tune_scatter_2d(data_red, keys_red, fig_name=f"{pb}_{noise_pow}_scatter2d")

        data_tv = data['data_tv']
        keys_tv = data['keys_tv']
        tune_plot_1d(data_tv, keys_tv, fig_name=f"{pb}_{noise_pow}_plot1d")

def main_lipschitz():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    denoiser = DRUNet(device=device)
    sigma_vec = [0.02 + n * 0.001 for n in range(0, 200)]

    #measure_lipschitz(denoiser, sigma_vec=sigma_vec, device=device, sigma_noise=0.1)
    #measure_lipschitz(denoiser, sigma_vec=sigma_vec, device=device, sigma_noise=0.2)

    # sigma_noise is None => denoiser match the true noise level
    measure_lipschitz(denoiser, sigma_vec=sigma_vec, device=device, sigma_noise=None)

if __name__ == "__main__":
    print(sys.prefix)
    # 1 perform grid search
    #main_tune(plot_and_exit=True)

    # CPROFILE
    #main_test('inpainting', test_dataset=False, benchmark=False, noise_pow=0.1)

    # 2 quick tests + benchmark
    #main_test('inpainting', test_dataset=False, benchmark=True, noise_pow=0.1)
    #main_test('blur', test_dataset=False, benchmark=True, noise_pow=0.1)
    main_test('tomography', test_dataset=False, noise_pow=0.3)
    #main_test('tomography', test_dataset=False, benchmark=True, noise_pow=0.3)

    #main_test('inpainting', dataset_name='celeba', nb_subset=30)

    # 3 database tests
    #main_test('blur', test_dataset=True)
    #main_test('inpainting', test_dataset=True)

    #main_test('tomography', test_dataset=False)

    #plot_spectr_ratio()
    #main_lipschitz()

    # test_rastrigin()
