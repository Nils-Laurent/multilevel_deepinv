import sys

import torch
from torch.utils.data import Subset, Dataset
from torchvision import transforms
from itertools import product

from gen_fig.fig_metric_logger import MRedMLInit, MRedInit, MRed, MRedML, MDPIR, MFb, MFbML, MPnPML, MPnP
from utils.measure_data import create_measure_data, load_measure_data

if "/.fork" in sys.prefix:
    sys.path.append('/projects/UDIP/nils_src/deepinv')

import matplotlib
matplotlib.use('module://backend_interagg')

import deepinv
from deepinv.physics import GaussianNoise
from deepinv.utils.demo import load_dataset
from tests.parameters import get_parameters_tv, get_parameters_red, get_parameters_pnp, single_level_params
from tests.test_alg import RunAlgorithm
from tests.utils import physics_from_exp, data_from_user_input, ResultManager
from utils.npy_utils import save_grid_tune_info, load_variables_from_npy, grid_search_npy_filename
from utils.gridsearch import tune_grid_all
from utils.gridsearch_plots import tune_scatter_2d, tune_plot_1d, print_gridsearch_max
from utils.paths import dataset_path, get_out_dir


def test_settings(data_in, params_exp, device, benchmark=False, physics=None, list_method=None):
    #print("Using torch.manual_seed")
    #params_exp["manual_seed"] = True

    noise_pow = params_exp["noise_pow"]
    print("def_noise:", noise_pow)

    tensor_np = torch.tensor(noise_pow).to(device)
    g = GaussianNoise(sigma=tensor_np)
    exp_physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, exp_physics, params_exp, problem_name, device)
    if physics is None:
        physics = exp_physics

    b_dataset = not(type(data_in) == torch.Tensor)
    rm = ResultManager(b_dataset=b_dataset)

    # ============== RED ==============
    p_red, param_init = get_parameters_red(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init, return_timer=benchmark)
    if MRedMLInit in list_method:
        rm.post_process(ra.RED_GD(p_red), MRedMLInit().key)

    p_red, param_init = get_parameters_red(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init, return_timer=benchmark)
    if MRedInit in list_method:
        rm.post_process(ra.RED_GD(single_level_params(p_red)), MRedInit().key)

    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    if MRedML in list_method:
        rm.post_process(ra.RED_GD(p_red.copy()), MRedML().key)
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    if MRed in list_method:
        rm.post_process(ra.RED_GD(single_level_params(p_red.copy())), MRed().key)

    # ============== PnP ==============
    p_pnp = get_parameters_pnp(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    if MPnPML in list_method:
        rm.post_process(ra.PnP_PGD(p_pnp.copy()), MPnPML().key)
    p_pnp = get_parameters_pnp(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    if MPnP in list_method:
        rm.post_process(ra.PnP_PGD(single_level_params(p_pnp.copy())), MPnP().key)

    # ============== DPIR ==============
    p_red, param_init = get_parameters_red(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    if MDPIR in list_method:
        rm.post_process(ra.DPIR(single_level_params(p_red.copy())), MDPIR().key)

    # ============== PGD ==============
    p_tv = get_parameters_tv(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    if MFb in list_method:
        rm.post_process(ra.TV_PGD(p_tv), MFbML().key)
    p_tv = get_parameters_tv(params_exp)
    ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark)
    if MFbML in list_method:
        rm.post_process(ra.TV_PGD(single_level_params(p_tv)), MFb().key)

    rm.finalize(list_method, params_exp, benchmark)


def main_test(
        problem,
        test_dataset=True,
        tune=False,
        benchmark=False,
        noise_pow=0.1,
        dataset_name='set3c',
        nb_subset=None,
        img_size=None,
        target=None,
        use_file_data=True,
        m_vec = None
):
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(device)

    original_data_dir = dataset_path()
    if img_size is None:
        val_transform = transforms.Compose([transforms.ToTensor()])
    else:
        val_transform = transforms.Compose([transforms.CenterCrop(img_size), transforms.ToTensor()])

    # inpainting: proportion of pixels to keep
    params_exp = {'problem': problem, 'set_name': dataset_name, 'shape': (3, img_size, img_size)}
    params_exp['noise_pow'] = noise_pow
    if problem == 'inpainting':
        params_exp[problem] = 0.5
    elif problem == 'tomography':
        params_exp[problem] = 0.6
    elif problem == 'blur':
        params_exp[problem + '_pow'] = 3.6
    else:
        raise NotImplementedError()

    physics = None
    if use_file_data:
        params_exp['online'] = False
        data, physics = load_measure_data(params_exp, device)
    else:
        params_exp['online'] = True
        dataset = load_dataset(dataset_name, original_data_dir, transform=val_transform)
        if nb_subset is not None:
            dataset = Subset(dataset, range(0, nb_subset))
        data = dataset

        if tune is True:
            return tune_grid_all(dataset, params_exp, device)

    if tune is True:
        return None

    with torch.no_grad():
        if test_dataset is True:
            test_settings(data, params_exp, device=device, benchmark=benchmark, physics=physics, list_method=m_vec)
        elif isinstance(data, Dataset):
            id_img = 0
            for t in data:
                id_img += 1
                if not (target is None) and id_img != target:
                    continue

                name_id = dataset_name + "_" + str(id_img)
                params_exp['img_name'] = name_id

                img = t[0].unsqueeze(0).to(device)
                test_settings(img, params_exp, device=device, benchmark=benchmark)


def main_tune(plot_and_exit=False):
    #pb_list = ['inpainting', 'blur', 'tomography']
    #noise_pow_vec = [0.01, 0.05, 0.1, 0.2, 0.3]
    pb_list = ['inpainting', 'blur']
    noise_pow_vec = [0.01, 0.1, 0.2]

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


def main_tune_plot(pb_list, noise_pow_vec):
    for pb, noise_pow in product(pb_list, noise_pow_vec):
        file_pb = grid_search_npy_filename(suffix=pb + str(noise_pow))
        data = load_variables_from_npy(file_pb)

        for key_ in data.keys():
            axis = data[key_]['axis']
            tensors = data[key_]['tensors']
            if len(axis) == 2:
                tune_scatter_2d(tensors, axis, fig_name=f"{pb}_{noise_pow}_{key_}_scatter2d")
                print_gridsearch_max(tensors, axis, f"{pb}_{noise_pow}_{key_}_scatter2d")
            else:
                tune_plot_1d(tensors, axis, fig_name=f"{pb}_{noise_pow}_{key_}_plot1d")
                print_gridsearch_max(tensors, axis, f"{pb}_{noise_pow}_{key_}_plot1d")


if __name__ == "__main__":
    print(sys.prefix)
    m_vec_red = [MRedMLInit, MRedInit, MRedML, MRed, MDPIR, MFb, MFbML]
    m_vec_pnp = [MPnPML, MPnP, MDPIR, MFb, MFbML]

    # 1 perform grid search
    #create_measure_data('inpainting', dataset_name='set3c', noise_pow=0.1, img_size=256)
    #create_measure_data('blur', dataset_name='set3c', noise_pow=0.1, img_size=256)
    main_test('blur', img_size=256, dataset_name='set3c', noise_pow=0.1, m_vec=m_vec_pnp, benchmark=True)
    #main_tune(plot_and_exit=False)
    #main_tune(plot_and_exit=True)

    # 2 quick tests + benchmark
    #main_test('inpainting', img_size=256, dataset_name='set3c', test_dataset=False,  noise_pow=0.1, target=2)
    #main_test('inpainting', img_size=1024, dataset_name='DIV2K', test_dataset=False, noise_pow=0.1, target=1)
    #main_test('blur', img_size=1024, dataset_name='DIV2K', test_dataset=False, benchmark=True, noise_pow=0.05)
    #main_test('tomography', dataset_name='DIV2K', test_dataset=False, benchmark=True, noise_pow=0.2)

    # 3 datasets
    #main_test('inpainting', img_size=256, noise_pow=0.1)
    #main_test('inpainting', img_size=256, benchmark=True, noise_pow=0.1)
    #main_test('inpainting', img_size=1024, dataset_name='DIV2K', noise_pow=0.1)
    #main_test('tomography', dataset_name='DIV2K', noise_pow=0.1)

    # FIG GUILLAUME : noise = 0.01, blur_pow = 3.6 ?
    #main_test('blur', img_size=2048, dataset_name='astro_ml', benchmark=True, test_dataset=False, noise_pow=0.01)
    #main_test('inpainting', img_size=2048, dataset_name='astro_ml', benchmark=True, test_dataset=False, noise_pow=0.01)
