import cProfile
import torch.profiler as profiler
import os
import sys
import torch
from torch.utils.data import Subset
from torchvision import transforms
from itertools import product

if "/.fork" in sys.prefix:
    sys.path.append('/projects/UDIP/nils_src/deepinv')

import deepinv
from deepinv.physics import GaussianNoise
from deepinv.utils.demo import load_dataset
from tests.parameters import get_parameters_tv, get_parameters_red
from tests.test_alg import RunAlgorithm
from tests.utils import physics_from_exp, data_from_user_input
from tests.utils import single_level_params
from utils.npy_utils import save_grid_tune_info, load_variables_from_npy, grid_search_npy_filename
from utils.gridsearch import tune_grid_all
from utils.gridsearch_plots import tune_scatter_2d, tune_plot_1d
from utils.paths import dataset_path, get_out_dir


def test_settings(data_in, params_exp, device, benchmark=False):
    if type(data_in) == torch.Tensor:
        print("Single image : using torch.manual_seed")
        params_exp["manual_seed"] = True

    noise_pow = params_exp["noise_pow"]
    print("def_noise:", noise_pow)

    tensor_np = torch.tensor(noise_pow).to(device)
    g = GaussianNoise(sigma=tensor_np)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    # ============== RED ==============
    p_red, param_init = get_parameters_red(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init, return_timer=benchmark)
    ra.RED_GD(p_red)
    p_red, param_init = get_parameters_red(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init, return_timer=benchmark)
    ra.RED_GD(single_level_params(p_red))

    #ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    #ra.RED_GD(p_red.copy())
    #ra.RED_GD(single_level_params(p_red.copy()))

    # ============== DPIR ==============
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    ra.DPIR(single_level_params(p_red.copy()))

    # ============== PGD ==============
    # /!\ sans torch.manual_seed, TV et TV multilevel n'ont pas la même cost car la réalisation de bruit est différente
    p_tv = get_parameters_tv(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    ra.TV_PGD(p_tv)
    p_tv = get_parameters_tv(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    ra.TV_PGD(single_level_params(p_tv))

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


def main_test(
        problem,
        test_dataset=True,
        tune=False,
        benchmark=False,
        noise_pow=0.1,
        dataset_name='set3c',
        nb_subset=None,
        img_size=None):
    #device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(device)

    original_data_dir = dataset_path()
    if img_size is None:
        img_size = 256 if torch.cuda.is_available() else 64

    val_transform = transforms.Compose(
        [transforms.CenterCrop(img_size), transforms.ToTensor()]
    )

    dataset = load_dataset(dataset_name, original_data_dir, transform=val_transform)
    if nb_subset is not None:
        dataset = Subset(dataset, range(0, nb_subset))

    # inpainting: proportion of pixels to keep
    params_exp = {'problem': problem, 'set_name': dataset_name, 'shape': (3, img_size, img_size)}
    params_exp['noise_pow'] = noise_pow
    if problem == 'inpainting':
        params_exp[problem] = 0.5
    elif problem == 'tomography':
        params_exp[problem] = 0.6
    elif problem == 'blur':
        params_exp[problem + '_pow'] = 2.0
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
        print(f"p_red = {data['res_red']}")
        print(f"p_tv = {data['res_tv']}")

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

if __name__ == "__main__":
    print(sys.prefix)
    # 1 perform grid search
    #main_tune(plot_and_exit=False)
    #main_test('inpainting', img_size=1024, dataset_name='astro_ml', test_dataset=False, benchmark=True, noise_pow=0.1)
    #main_test('blur', img_size=512, dataset_name='astro_ml', test_dataset=False, benchmark=True, noise_pow=0.05)
    # FIG GUILLAUME : blur_pow = 4.0, noise = 0.01, hyper params noise 0.05,
    main_test('blur', img_size=2048, dataset_name='astro_ml', benchmark=True, test_dataset=False, noise_pow=0.05)

    # CPROFILE
    #main_test('inpainting', test_dataset=False, benchmark=False, noise_pow=0.1)

    # 2 quick tests + benchmark
    #main_test('inpainting', test_dataset=False, benchmark=True, noise_pow=0.1)
    #main_test('inpainting', test_dataset=False, benchmark=True, noise_pow=0.1)
    #main_test('blur', test_dataset=False, benchmark=True, noise_pow=0.1)
    #main_test('tomography', test_dataset=False, noise_pow=0.2)
    #main_test('tomography', test_dataset=False, benchmark=True, noise_pow=0.3)

    #main_test('inpainting', dataset_name='celeba', nb_subset=30)

    # 3 database tests
    #main_test('blur', test_dataset=True)
    #main_test('inpainting', test_dataset=True)

    #main_test('tomography', test_dataset=False)

    # test_rastrigin()
