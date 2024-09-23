import sys

#if "/.fork" in sys.prefix:
sys.path.append('/projects/UDIP/nils_src/deepinv')

from os.path import join

import numpy
import torch
import torchvision.transforms
from deepinv.datasets import FastMRISliceDataset
from deepinv.models import DRUNet
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms
from itertools import product

from utils.transforms import CatZeroChannel

from multilevel.info_transfer import BlackmannHarris, SincFilter, CFir, Dirac
from tests.parameters import get_multilevel_init_params, ConfParam
from utils.ml_dataclass import *

#from gen_fig.fig_metric_logger import *

from utils.measure_data import create_measure_data, load_measure_data

#import matplotlib
#matplotlib.use('module://backend_interagg')

import deepinv
from deepinv.physics import GaussianNoise
from deepinv.utils.demo import load_dataset
from tests.test_alg import RunAlgorithm
from tests.utils import physics_from_exp, data_from_user_input, ResultManager
from utils.npy_utils import save_grid_tune_info, load_variables_from_npy, grid_search_npy_filename
from utils.gridsearch import tune_grid_all
from utils.gridsearch_plots import tune_scatter_2d, tune_plot_1d, print_gridsearch_max
from utils.paths import dataset_path, get_out_dir


def test_settings(data_in, params_exp, device, benchmark=False, physics=None, list_method=None):
    if isinstance(data_in, torch.Tensor):
        print("Using torch.manual_seed")
        params_exp["manual_seed"] = True

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

    for m_class in list_method:
        m_param = m_class.param_fn(params_exp)
        ra = RunAlgorithm(data, physics, params_exp, device=device, return_timer=benchmark, def_name=m_class().key)
        if hasattr(m_class, 'use_init') and m_class.use_init is True:
            init_param = get_multilevel_init_params(m_param)
            ra.set_init(init_param)

        rm.post_process(ra.run_algorithm(m_class, m_param), m_class)

    rm.finalize(list_method, params_exp, benchmark)


def main_test(
        problem,
        test_dataset=True,
        tune=False,
        benchmark=False,
        noise_pow=0.1,
        dataset_name='set3c',
        subset_size=None,
        img_size=None,
        target=None,
        use_file_data=True,
        m_vec=None,
        cpu=False,
        device = "cpu",
):
    if cpu is True:
        device = "cpu"
    print(f"=== device : {device} ===")

    original_data_dir = dataset_path()
    transform_vec = [transforms.ToTensor()]
    if not(img_size is None) :
        if type(img_size) is int:
            if problem == 'mri':
                img_size = (2, img_size, img_size)
            else:
                img_size = (3, img_size, img_size)

            t_crop = (img_size[1], img_size[2])
            transform_vec = [transforms.CenterCrop(t_crop), transforms.ToTensor(), ]

    # inpainting: proportion of pixels to keep
    params_exp = {'problem': problem, 'set_name': dataset_name, 'shape': img_size, 'device': device}
    params_exp['noise_pow'] = noise_pow
    if problem == 'inpainting':
        params_exp[problem] = 0.5
    elif problem == 'tomography':
        params_exp[problem] = 0.6
    elif problem == 'blur':
        params_exp[problem + '_pow'] = 3.6
        #params_exp[problem + '_pow'] = 7.3
    elif problem == 'demosaicing' or problem == "motion_blur" or problem == "mri":
        # nothing to be done
        pass
    else:
        raise NotImplementedError()

    physics = None
    if use_file_data:
        params_exp['online'] = False
        data, physics = load_measure_data(params_exp, device, subset_size=subset_size, target=target)
    else:
        params_exp['online'] = True
        if dataset_name == 'knee_singlecoil':
            root = original_data_dir/dataset_name/"singlecoil_val"
            val_transform = transforms.Compose(transform_vec)
            dataset = deepinv.datasets.FastMRISliceDataset(root=root, test=False, challenge="singlecoil",
                transform_target=val_transform, #transform_kspace=val_transform
            )
        else:
            if problem == 'mri':
                transform_vec.append(transforms.Grayscale(num_output_channels=1))
                transform_vec.append(CatZeroChannel())
            val_transform = transforms.Compose(transform_vec)
            dataset = load_dataset(dataset_name, original_data_dir, transform=val_transform)
        if subset_size is not None:
            dataset = Subset(dataset, range(0, subset_size))
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
                if not (target is None):
                    if id_img < target:
                        id_img += 1
                        continue
                    elif id_img > target:
                        break

                name_id = dataset_name + "_" + str(id_img)
                params_exp['img_name'] = name_id

                if isinstance(t, tuple):
                    img = t[0].unsqueeze(0).to(device)
                if params_exp['problem'] == "mri":
                    img = img/torch.max(img)
                test_settings(img, params_exp, device=device, benchmark=benchmark, list_method=m_vec)
                id_img += 1


def main_tune(device, plot_and_exit=False):
    #noise_pow_vec = [0.05, 0.1, 0.2]
    #pb_list = ['inpainting', 'blur']
    pb_list = ['inpainting', 'demosaicing', 'blur', 'mri']
    pb_list = ['demosaicing', 'blur', 'mri']
    pb_list = ['mri']
    noise_pow_vec = [0.1]

    noise_pow_vec = numpy.sort(noise_pow_vec)
    if plot_and_exit is True:
        main_tune_plot(pb_list, noise_pow_vec)
        return

    for pb, noise_pow in product(pb_list, noise_pow_vec):
        if pb == 'blur':
            conf_param = ConfParam()
            conf_param.levels = 2
            conf_param.iters_fine = 200
            conf_param.iml_max_iter = 5
        if pb == 'mri':
            conf_param = ConfParam()
            conf_param.win = SincFilter()
            conf_param.levels = 3
            conf_param.iters_fine = 200
            conf_param.iml_max_iter = 8
            conf_param.use_complex_denoiser = True
            conf_param.denoiser_in_channels = 1  # separated real and imag parts
            conf_param.coarse_iters_ini = 4
        r_pb = main_test(pb, dataset_name='gridsearch', img_size=1024, noise_pow=noise_pow,
                         tune=True, use_file_data=False, device=device)
        file_pb = save_grid_tune_info(data=r_pb, suffix=pb + str(noise_pow))

    print("_____________________________")
    print("grid search finished")
    print("_____________________________")
    for pb, noise_pow in product(pb_list, noise_pow_vec):
        file_pb = grid_search_npy_filename(pb + str(noise_pow))
        data = load_variables_from_npy(file_pb)


def main_tune_plot(pb_list, in_noise_pow_vec):
    noise_pow_vec = numpy.sort(in_noise_pow_vec)
    for pb, noise_pow in product(pb_list, noise_pow_vec):
        file_pb = grid_search_npy_filename(suffix=pb + str(noise_pow))
        data = load_variables_from_npy(file_pb)

        for key_ in data.keys():
            axis = data[key_]['axis']
            tensors = data[key_]['tensors']
            if len(axis) == 2:
                tune_scatter_2d(tensors, axis, fig_name=f"{pb}_{noise_pow}_{key_}_scatter2d")
            else:
                tune_plot_1d(tensors, axis, fig_name=f"{pb}_{noise_pow}_{key_}_plot1d")

            print_gridsearch_max(key_, tensors, axis, noise_pow)

def main_fn():
    print(sys.prefix)
    conf_param = ConfParam()
    #set3c_shape = (3, 256, 256)
    #div2k_shape = (3, 2040, 1356)
    set3c_sz = 256
    div2k_sz = 1024
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    #x = torch.rand(1, 3, 64, 65).to(device)
    #denoiser = DRUNet(pretrained="download", device=device)
    #denoiser.eval()
    #sigma = 0.05
    #y = denoiser(x, sigma)

    #methods_noreg = [
    #    MPnP, MPnPML, MPnPMLStudNoR, MPnPMoreau,
    #    MPnPProx, MPnPProxML, MPnPProxMLStudNoR, MPnPProxMoreau,
    #    MFb, MFbMLGD,
    #    MRed, MRedML, MRedMLStudNoR, MRedMLMoreau,
    #    MDPIR,
    #]
    #methods_noreg_init = [
    #    MPnP, MPnPML, MPnPMLInit, MPnPMLStudNoRInit, MPnPMoreauInit,
    #    MPnPProx, MPnPProxMLInit, MPnPProxMLStudNoRInit, MPnPProxMoreauInit,
    #    MFb, MFbMLGD,
    #    MRed, MRedMLInit,MRedMLStudNoRInit, MRedMLMoreauInit,
    #    MDPIR,
    #]
    methods_standard = [
        #MPnP, MPnPML, MPnPMLStud, MPnPMoreau,
        MPnPProx, MPnPProxML, MPnPProxMLStud, MPnPProxMoreau,
        MFb, MFbMLGD,
        MRed, MRedML,MRedMLStud, MRedMLMoreau,
        MDPIR,
    ]
    methods_init = [
        #MPnP, MPnPInit, MPnPML, MPnPMLInit, MPnPMLStud, MPnPMLStudInit, MPnPMoreau, MPnPMoreauInit,
        MPnPProx, MPnPProxInit, MPnPProxML, MPnPProxMLInit, MPnPProxMLStud, MPnPProxMLStudInit, MPnPProxMoreau, MPnPProxMoreauInit,
        MFb, MFbMLGD,
        MRed, MRedInit, MRedML, MRedMLInit, MRedMLStudInit, MRedMLMoreau, MRedMLMoreauInit,
        MDPIR,
    ]

    # 1 create degraded datasets
    #create_measure_data('blur', dataset_name='set3c', noise_pow=0.01, img_size=set3c_shape)
    #create_measure_data('blur', dataset_name='set3c', noise_pow=0.1, img_size=set3c_shape)
    #create_measure_data('blur', dataset_name='set3c', noise_pow=0.2, img_size=set3c_shape)
    #create_measure_data('blur', dataset_name='DIV2K', noise_pow=0.01, img_size=div2k_shape)
    #create_measure_data('blur', dataset_name='DIV2K', noise_pow=0.1, img_size=div2k_shape)
    #create_measure_data('blur', dataset_name='DIV2K', noise_pow=0.2, img_size=div2k_shape)

    # 2 perform grid search
    conf_param.win = BlackmannHarris()
    conf_param.levels = 4
    conf_param.iters_fine = 200
    conf_param.iml_max_iter = 8
    conf_param.iml_max_iter = 1
    main_tune(device=device, plot_and_exit=False)
    main_tune(device=device, plot_and_exit=True)
    return None

    # 3 evaluate methods on single image
    # e.g. windows for downsampling CFir(), BlackmannHarris(), SincFilter()

    # -- inpainting ----------------------------------------------------------------
    methods_init = [MPnPML]
    conf_param.win = BlackmannHarris()
    conf_param.levels = 4
    conf_param.iters_fine = 200
    conf_param.iml_max_iter = 8
    #main_test(
    #    'inpainting', img_size=1024, dataset_name='DIV2K', noise_pow=0.1, m_vec=methods_init, test_dataset=False,
    #    target=4, use_file_data=False, benchmark=True, cpu=False, device=device
    #)

    # -- demosaicing ----------------------------------------------------------------
    conf_param.win = BlackmannHarris()
    conf_param.levels = 4
    conf_param.iters_fine = 200
    conf_param.iml_max_iter = 8
    #main_test(
    #    'demosaicing', img_size=1024, dataset_name='DIV2K', noise_pow=0.1, m_vec=methods_init, test_dataset=False,
    #    target=4, use_file_data=False, benchmark=True, cpu=False, device=device
    #)

    # -- motion blur ----------------------------------------------------------------
    conf_param.win = BlackmannHarris()
    conf_param.levels = 2
    conf_param.iters_fine = 200
    conf_param.iml_max_iter = 5
    #main_test(
    #    'motion_blur', img_size=1024, dataset_name='DIV2K', noise_pow=0.1, m_vec=methods_standard, test_dataset=False,
    #    target=0, use_file_data=False, benchmark=True, cpu=False, device=device
    #)


    # -- MRI ----------------------------------------------------------------
    #conf_param.win = Dirac()
    #conf_param.win = BlackmannHarris()
    conf_param.win = SincFilter()
    conf_param.levels = 3
    conf_param.iters_fine = 200
    conf_param.iml_max_iter = 8
    conf_param.use_complex_denoiser = True
    conf_param.denoiser_in_channels = 1  # separated real and imag parts
    conf_param.coarse_iters_ini = 4
    methods_init_mri = [
        MPnP, MPnPInit, MPnPML, MPnPMLInit, MPnPMoreau, MPnPMoreauInit,
        MRed, MRedInit, MRedML, MRedMLInit, MRedMLMoreau, MRedMLMoreauInit,# MRedMLStudInit,
        MFb, MFbMLGD,
        MDPIR,
    ]
    #methods_init_mri = [MRedInit]
    #main_test(
    #    'mri', img_size=256, dataset_name='set3c', noise_pow=0.1, m_vec=methods_init_mri, test_dataset=False,
    #    target=1, use_file_data=False, benchmark=True, cpu=False, device=device
    #)

    img_size = (2, 320, 320)
    #main_test(
    #    'mri', img_size=img_size, dataset_name='DIV2K', noise_pow=0.1, m_vec=methods_init_mri, test_dataset=False,
    #    target=0, use_file_data=False, benchmark=True, cpu=False, device=device
    #)
    main_test(
        'mri', img_size=img_size, dataset_name='knee_singlecoil', noise_pow=0.1, m_vec=methods_init_mri, test_dataset=False,
        target=15, use_file_data=False, benchmark=True, cpu=False, device=device
    )

    # -- inpainting : set3c ----------------------------------------------------------------
    #main_test(
    #    'inpainting', img_size=256, dataset_name='set3c', noise_pow=0.01, m_vec=methods_noreg_init, test_dataset=False,
    #    target=1, use_file_data=False, benchmark=True, cpu=False, device=device
    #)
    #main_test(
    #    'inpainting', img_size=256, dataset_name='set3c', noise_pow=0.1, m_vec=methods_init, test_dataset=False,
    #    target=1, use_file_data=False, benchmark=True, cpu=False, device=device
    #)

    # -- blur ----------------------------------------------------------------
    # e.g. windows for downsampling CFir(), BlackmannHarris()
    #conf_param.win = BlackmannHarris()
    #conf_param.levels = 2
    #conf_param.iters_fine = 400
    #conf_param.iml_max_iter = 1
    #main_test(
    #    'blur', img_size=1024, dataset_name='DIV2K', noise_pow=0.1, m_vec=methods_standard, test_dataset=False,
    #    target=6, use_file_data=False, benchmark=True, cpu=False, device=device
    #)
    #main_test(
    #    'blur', img_size=1024, dataset_name='DIV2K', noise_pow=0.01, m_vec=methods_noreg, test_dataset=False,
    #    target=6, use_file_data=False, benchmark=True, cpu=False, device=device
    #)

    # 4 statistical tests
    #main_test(
    #    'blur', img_size=set3c_sz, dataset_name='set3c', noise_pow=0.1, m_vec=m_vec_pnp, test_dataset=True,
    #    use_file_data=True, benchmark=True, cpu=False
    #)

    # FIG GUILLAUME : noise = 0.01, blur_pow = 3.6 ?
    #main_test('blur', img_size=2048, dataset_name='astro_ml', benchmark=True, test_dataset=False, noise_pow=0.01)
    #main_test('inpainting', img_size=2048, dataset_name='astro_ml', benchmark=True, test_dataset=False, noise_pow=0.01)

if __name__ == "__main__":
    main_fn()