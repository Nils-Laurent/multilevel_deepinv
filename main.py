import sys
import torch
from torch.utils.data import Subset
from torchvision import transforms
from itertools import product

from gen_fig.fig_metric_logger import GenFigMetricLogger, MRedMLInit, MRedInit, MRed, MRedML, MDPIR, MFb, MFbML

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
from utils.paths import dataset_path


def test_settings(data_in, params_exp, device, benchmark=False):
    print("Using torch.manual_seed")
    params_exp["manual_seed"] = True

    noise_pow = params_exp["noise_pow"]
    print("def_noise:", noise_pow)

    tensor_np = torch.tensor(noise_pow).to(device)
    g = GaussianNoise(sigma=tensor_np)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    z = GenFigMetricLogger()

    # ============== RED ==============
    p_red, param_init = get_parameters_red(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init, return_timer=benchmark)
    z.add_logger(ra.RED_GD(p_red), MRedMLInit().key)

    p_red, param_init = get_parameters_red(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init, return_timer=benchmark)
    z.add_logger(ra.RED_GD(single_level_params(p_red)), MRedInit().key)

    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    z.add_logger(ra.RED_GD(p_red.copy()), MRedML().key)
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    z.add_logger(ra.RED_GD(single_level_params(p_red.copy())), MRed().key)

    # ============== DPIR ==============
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    z.add_logger(ra.DPIR(single_level_params(p_red.copy())), MDPIR().key)

    # ============== PGD ==============
    p_tv = get_parameters_tv(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    z.add_logger(ra.TV_PGD(p_tv), MFbML().key)
    p_tv = get_parameters_tv(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    z.add_logger(ra.TV_PGD(single_level_params(p_tv)), MFb().key)

    z.gen_fig('psnr')


def main_test(
        problem,
        test_dataset=True,
        tune=False,
        benchmark=False,
        noise_pow=0.1,
        dataset_name='set3c',
        nb_subset=None,
        img_size=None):
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(device)

    original_data_dir = dataset_path()
    if img_size is None:
        val_transform = transforms.Compose([transforms.ToTensor()])
    else:
        val_transform = transforms.Compose([transforms.CenterCrop(img_size), transforms.ToTensor()])

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
        params_exp[problem + '_pow'] = 4.0
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
            #if id_img == 1:
            #    continue
            name_id = dataset_name + "_" + str(id_img)
            params_exp['img_name'] = name_id

            img = t[0].unsqueeze(0).to(device)
            test_settings(img, params_exp, device=device, benchmark=benchmark)
            break


def main_tune(plot_and_exit=False):
    #pb_list = ['inpainting', 'blur', 'tomography']
    pb_list = ['inpainting', 'blur']
    #noise_pow_vec = [0.05, 0.1, 0.2, 0.3]
    noise_pow_vec = [0.2]

    if plot_and_exit is True:
        main_tune_plot(pb_list, noise_pow_vec)
        return

    for pb, noise_pow in product(pb_list, noise_pow_vec):
        r_pb = main_test(pb, dataset_name='set3c', img_size=256, noise_pow=noise_pow, tune=True)
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
    #main_tune(plot_and_exit=True)

    # 2 quick tests + benchmark
    #main_test('inpainting', img_size=1024, dataset_name='DIV2K', test_dataset=False, benchmark=True, noise_pow=0.05)
    #main_test('blur', img_size=1024, dataset_name='DIV2K', test_dataset=False, benchmark=True, noise_pow=0.05)
    #main_test('tomography', dataset_name='DIV2K', test_dataset=False, benchmark=True, noise_pow=0.2)

    # 3 database tests
    #main_test('blur', img_size=256, benchmark=True, noise_pow=0.1)
    #main_test('blur', dataset_name='DIV2K', noise_pow=0.1)
    main_test('inpainting', dataset_name='DIV2K', noise_pow=0.1)
    #main_test('tomography', dataset_name='DIV2K', noise_pow=0.1)

    # FIG GUILLAUME : blur_pow = 4.0, noise = 0.01, hyper params noise 0.05,
    #main_test('blur', img_size=2048, dataset_name='astro_ml', benchmark=True, test_dataset=False, noise_pow=0.01)
    #main_test('inpainting', img_size=2048, dataset_name='astro_ml', benchmark=True, test_dataset=False, noise_pow=0.05)
