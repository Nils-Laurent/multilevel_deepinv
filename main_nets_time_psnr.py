import sys

from main import main_test

#if "/.fork" in sys.prefix:
sys.path.append('/projects/UDIP/nils_src/deepinv')

import torch

from utils.parameters import ConfParam
from utils.ml_dataclass import *
from utils.ml_dataclass_denoiser import *
from utils.parameters_global import FixedParams

import deepinv
from utils.run_alg import RunAlgorithm

def main_nets_time_psnr():
    print(sys.prefix)
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    methods_base = [
        MPnP, MPnPInit, MPnPML, MPnPMLInit, MPnPMoreau, MPnPMoreauInit,
        MFb, MFbMLGD,
        MDPIR, MDPIRLong,
    ]
    methods_alt = [MPnPSCUNet, MPnPMLSCUNetInit, MPnPProx, MPnPProxMLInit]

    methods_init = methods_base + methods_alt
    methods_img = []
    RunAlgorithm.class_vec_save_img = methods_img

    ConfParam().reset()
    ConfParam().inpainting_ratio = 0.5  # keep 50%
    for step_sz in [0.1, 0.3, 0.6, 0.9]:
        for g_par in [0.02, 0.05, 0.08, 0.11]:
            FixedParams().reset()
            FixedParams().g_param = g_par
            FixedParams().stepsize_coeff = step_sz
            methods_test = [MPnP, MPnPMLInit, MPnPProx, MPnPProxMLInit]
            main_test(
                'inpainting', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_test, test_dataset=False,
                use_file_data=False, benchmark=True, cpu=False, device=device, target=3
            )
    for step_sz in [0.1, 0.3, 0.6, 0.9]:
        FixedParams().stepsize_coeff = step_sz
        methods_test = [MPnPDnCNN, MPnPMLDnCNNInit, MPnPSCUNet, MPnPMLSCUNetInit]
        main_test(
            'inpainting', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_test, test_dataset=False,
            use_file_data=False, benchmark=True, cpu=False, device=device, target=3
        )
        FixedParams().reset()

    #ConfParam().reset()
    #for step_sz in [0.8, 0.9, 1.0, 1.1, 1.2]:
    #    FixedParams().stepsize_coeff = step_sz
    #    methods_test = [MPnP, MPnPMLInit, MPnPProx, MPnPProxMLInit]
    #    main_test(
    #        'demosaicing', img_size=1024, dataset_name='cset', noise_pow=0.1, m_vec=methods_test, test_dataset=False,
    #        use_file_data=False, benchmark=True, cpu=False, device=device, target=3
    #    )
    #    FixedParams().reset()
    return None