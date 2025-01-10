import sys

#if "/.fork" in sys.prefix:
sys.path.append('/projects/UDIP/nils_src/deepinv')

import numpy
import torch
from torch.utils.data import Subset, Dataset
from torchvision import transforms
from itertools import product

from utils.transforms import CatZeroChannel

from utils.parameters import ConfParam
from utils.ml_dataclass import *
from utils.ml_dataclass_denoiser import *
from utils.ml_dataclass_nonexp import *
from utils.parameters_global import FixedParams

from utils.measure_data import load_measure_data

#matplotlib.use('module://backend_interagg')

import deepinv
from deepinv.physics import GaussianNoise
from deepinv.utils.demo import load_dataset
from utils.run_alg import RunAlgorithm
from utils.utils import physics_from_exp, data_from_user_input, ResultManager
from utils.npy_utils import save_grid_tune_info, load_variables_from_npy, grid_search_npy_filename
from utils.gridsearch import tune_grid_all
from utils.gridsearch_plots import tune_scatter_2d, tune_plot_1d, print_gridsearch_max
from utils.paths import dataset_path

from multilevel_utils.custom_poisson_noise import CPoissonNoise, CPoissonLikelihood
from utils.parameters_utils import set_multilevel_init_params


def test_settings(data_in, params_exp, device, benchmark=False, physics=None, list_method=None):
    if isinstance(data_in, torch.Tensor):
        print("Using torch.manual_seed")
        params_exp["manual_seed"] = True

    noise_pow = params_exp["noise_pow"]
    print("def_noise:", noise_pow)

    tensor_np = torch.tensor(noise_pow).to(device)
    if isinstance(ConfParam().data_fidelity(), CPoissonLikelihood):
        bkg = ConfParam().data_fidelity().bkg
        gain = ConfParam().data_fidelity().gain
        g = CPoissonNoise(gain=gain)
    else:
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
            set_multilevel_init_params(m_param)
            #init_param = set_multilevel_init_params(m_param)
            #ra.set_init(init_param)

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
        params_exp[problem] = ConfParam().inpainting_ratio
    elif problem == 'tomography':
        params_exp[problem] = 0.6
    elif problem == 'blur':
        params_exp[problem + '_pow'] = 3.6
    elif problem == 'demosaicing' or problem == "motion_blur" or problem == "mri" or problem == "denoising":
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
                    else:
                        try:
                            print(f"reach target: {data.samples[id_img][0]}")
                        except:
                            pass

                name_id = dataset_name + "_" + str(id_img)
                params_exp['img_name'] = name_id

                if isinstance(t, tuple):
                    img = t[0].unsqueeze(0).to(device)
                if params_exp['problem'] == "mri":
                    img = img/torch.max(img)
                test_settings(img, params_exp, device=device, benchmark=benchmark, list_method=m_vec)
                id_img += 1


def main_tune(device, plot_and_exit=False):
    #pb_list = ['inpainting', 'demosaicing', 'blur', 'mri']
    #pb_list = ['inpainting', 'demosaicing', 'blur', 'denoising']
    pb_list = ['denoising']
    noise_pow_vec = [0.1]

    noise_pow_vec = numpy.sort(noise_pow_vec)
    if plot_and_exit is True:
        main_tune_plot(pb_list, noise_pow_vec)
        return

    for pb, noise_pow in product(pb_list, noise_pow_vec):
        ConfParam().reset()
        if pb == 'denoising':
            bkg = 1
            gain = 1/10
            ConfParam().data_fidelity = lambda: CPoissonLikelihood(gain=gain, bkg=bkg, denormalize=True)
            ConfParam().data_fidelity_lipschitz = 1/(gain*bkg)**2
        if pb == 'mri':
            ConfParam().levels = 3
            ConfParam().coarse_iters_ini = 1
            ConfParam().use_complex_denoiser = True
            ConfParam().denoiser_in_channels = 1  # separated real and imag parts
        #dataset_name = 'gridsearch'  # high resolution images
        #img_size = 1024
        dataset_name = 'set3c'  # fast (small images)
        img_size = 256
        r_pb = main_test(pb, dataset_name=dataset_name, img_size=img_size, noise_pow=noise_pow,
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
        print(f"pb = {pb}")
        file_pb = grid_search_npy_filename(suffix=pb + str(noise_pow))
        data = load_variables_from_npy(file_pb)

        for key_ in data.keys():
            axis = data[key_]['axis']
            tensors = data[key_]['tensors']
            if len(axis) == 2:
                tune_scatter_2d(tensors, axis, fig_name=f"{pb}_{noise_pow}_{key_}_scatter2d")
            elif len(axis) == 1:
                tune_plot_1d(tensors, axis, fig_name=f"{pb}_{noise_pow}_{key_}_plot1d")

            print_gridsearch_max(key_, tensors, axis, noise_pow)

def main_fn():
    print(sys.prefix)
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    main_tune(device=device, plot_and_exit=False)
    main_tune(device=device, plot_and_exit=True)
    return None

    methods_base = [
        MPnP, MPnPInit, MPnPML, MPnPMLInit, MPnPMoreau, MPnPMoreauInit,
        MFb, MFbMLGD,
        MDPIR, MDPIRLong,
    ]
    methods_alt = [MPnPDnCNN, MPnPMLDnCNNInit, MPnPSCUNet, MPnPMLSCUNetInit, MPnPProx, MPnPProxMLInit]
    methods_alt_moreau = [MPnPMLDnCNNMoreauInit, MPnPMLSCUNetMoreauInit, MPnPProxMoreauInit]
    methods_ne = [MPnPNE, MPnPNEInit, MPnPNEML, MPnPNEMLInit, MPnPNEMoreau, MPnPNEMoreauInit]

    methods_init = methods_base + methods_alt + methods_alt_moreau
    methods_img = [MPnP, MPnPMLInit, MPnPMLDnCNNInit, MPnPMLSCUNetInit, MPnPProxMLInit, MDPIRLong, MFbMLGD, MPnPProx]
    RunAlgorithm.class_vec_save_img = methods_img

    ## -- Poisson ----------------------------------------------------------------
    ConfParam().reset()
    ConfParam().iters_fine = 200
    #ConfParam().iml_max_iter = 4

    bkg = 1
    gain = 1/10
    ConfParam().data_fidelity = lambda: CPoissonLikelihood(gain=gain, bkg=bkg, denormalize=True)
    ConfParam().data_fidelity_lipschitz = 1/(gain*bkg)**2
    ConfParam().use_equivariance = True

    methods_init_pl = methods_init
    main_test(
        'denoising', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_init_pl, test_dataset=False,
        use_file_data=False, benchmark=True, cpu=False, device=device, target=3
    )
    return None

    #methods_init = methods_prox
    # -- inpainting ----------------------------------------------------------------
    ConfParam().reset()
    ConfParam().inpainting_ratio = 0.5  # keep 50%
    for step_sz in [0.1, 0.3, 0.6, 0.9]:
        for g_par in [0.02, 0.05, 0.08, 0.11]:
            FixedParams().g_param = g_par
            FixedParams().stepsize_coeff = step_sz
            methods_test = [MPnP, MPnPMLInit, MPnPProx, MPnPProxMLInit]
            main_test(
                'inpainting', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_test, test_dataset=False,
                use_file_data=False, benchmark=True, cpu=False, device=device, target=3
            )
            FixedParams().reset()
    for step_sz in [0.1, 0.3, 0.6, 0.9]:
        FixedParams().stepsize_coeff = step_sz
        methods_test = [MPnPDnCNN, MPnPMLDnCNNInit, MPnPSCUNet, MPnPMLSCUNetInit]
        main_test(
            'inpainting', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_test, test_dataset=False,
            use_file_data=False, benchmark=True, cpu=False, device=device, target=3
        )
        FixedParams().reset()

    return None
    ConfParam().reset()
    ConfParam().inpainting_ratio = 0.5  # keep 50%

    main_test(
        'inpainting', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_init, test_dataset=False,
        use_file_data=False, benchmark=True, cpu=False, device=device, target=3
    )
    #return None
    #ConfParam().reset()
    #ConfParam().inpainting_ratio = 0.8  # keep 80%
    #main_test(
    #    'inpainting', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_init, test_dataset=False,
    #    use_file_data=False, benchmark=True, cpu=False, device=device
    #)
    ##return None
    #ConfParam().reset()
    #ConfParam().inpainting_ratio = 0.9  # keep 90%
    #main_test(
    #    'inpainting', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_init, test_dataset=False,
    #    use_file_data=False, benchmark=True, cpu=False, device=device
    #)
    ##return None

    # -- demosaicing ----------------------------------------------------------------
    #ConfParam().reset()
    #for step_sz in [0.8, 0.9, 1.0, 1.1, 1.2]:
    #    FixedParams().stepsize_coeff = step_sz
    #    methods_test = [MPnP, MPnPMLInit, MPnPProx, MPnPProxMLInit]
    #    main_test(
    #        'demosaicing', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_test, test_dataset=False,
    #        use_file_data=False, benchmark=True, cpu=False, device=device, target=3
    #    )
    #    FixedParams().reset()
    #return None

    ConfParam().reset()
    main_test(
        'demosaicing', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_init, test_dataset=False,
        use_file_data=False, benchmark=True, cpu=False, device=device, target=3
    )
    return None

    # -- blur ----------------------------------------------------------------
    ConfParam().reset()
    main_test(
        'blur', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_init, test_dataset=False,
        use_file_data=False, benchmark=True, cpu=False, device=device
    )
    return None

    # -- MRI ----------------------------------------------------------------
    ConfParam().reset()
    ConfParam().levels = 3
    ConfParam().coarse_iters_ini = 1
    ConfParam().use_complex_denoiser = True
    ConfParam().denoiser_in_channels = 1  # separated real and imag parts
    methods_init_mri = [
        MPnP, MPnPInit, MPnPML, MPnPMLInit, MPnPMoreau, MPnPMoreauInit,
        MFb, MFbMLGD,
        MDPIR, MDPIRLong,
    ]

    ConfParam().s1coherent_algorithm = True
    main_test(
        'mri', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_init_mri, test_dataset=False,
        use_file_data=False, benchmark=True, cpu=False, device=device
    )
    return None

    # -- motion blur ----------------------------------------------------------------
    methods_standard = [
        MPnP, MPnPML, MPnPMoreau,
        MPnPProx, MPnPProxML, MPnPProxMoreau,
        MFb, MFbMLGD,
        MRed, MRedML,MRedMLMoreau,
        MDPIR,
    ]
    # something is strange here
    # when win = SincFilter, the algorithm is very slow
    # when using BlackmannHarris it is normal
    # the probem also seems very difficult from multilevel perspective
    #ConfParam().reset()
    #ConfParam().levels = 3
    #methods_init = [MPnPML, MPnPMoreau]
    #main_test(
    #    'motion_blur', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_init, test_dataset=False,
    #    target=0, use_file_data=False, benchmark=True, cpu=False, device=device
    #)


    # 1 create degraded datasets
    #create_measure_data('blur', dataset_name='...', noise_pow=0.2, img_size=div2k_shape)

    # 2 perform grid search
    #main_tune(device=device, plot_and_exit=False)
    #main_tune(device=device, plot_and_exit=True)
    #return None

    # 3 evaluate methods on single image
    # e.g. windows for downsampling CFir(), BlackmannHarris(), SincFilter()

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