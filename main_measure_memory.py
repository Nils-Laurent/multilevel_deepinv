import sys

#if "/.fork" in sys.prefix:
sys.path.append('/projects/UDIP/nils_src/deepinv')

from deepinv.physics.blur import gaussian_blur

from multilevel_utils.custom_blur import CBlur

import deepinv
import torch
from scipy.io import savemat
from torchvision import transforms
from deepinv.models import EquivariantDenoiser, DRUNet
from deepinv.optim import optim_builder, PnP, L2
from deepinv.optim.optim_iterators import PGDIteration
from deepinv.physics import Inpainting, Demosaicing, Blur, GaussianNoise
from deepinv.utils.demo import load_dataset

from multilevel.coarse_pgd import CPGDIteration
from multilevel.info_transfer import SincFilter
from multilevel.iterator import MultiLevelIteration
from utils.paths import dataset_path


def get_physics(shape, pb, device):
    if pb == "inpainting":
        noise_level = torch.tensor(0.1)
        noise_model = GaussianNoise(sigma=noise_level)
        physics = Inpainting(shape, mask=0.5, noise_model=noise_model, device=device)
    elif pb == "demosaicing":
        noise_level = torch.tensor(0.1)
        noise_model = GaussianNoise(sigma=noise_level)
        physics = Demosaicing(shape, noise_model=noise_model, device=device)
    elif pb == "blur":
        noise_level = torch.tensor(0.1)
        noise_model = GaussianNoise(sigma=noise_level)
        power = 3.6
        physics = CBlur(gaussian_blur(sigma=(power, power), angle=0), noise_model=noise_model, device=device, padding='replicate')
    return physics


def get_img(pb, device, id_img):
    shape = (3, 1024, 1024)  # rgb 1024x1024 pixels
    size = (1024, 1024)  # rgb 1024x1024 pixels

    tr_vec = transforms.Compose([transforms.CenterCrop(size), transforms.ToTensor(), ])
    dataset = load_dataset('LIU4K-v2', dataset_path(), transform=tr_vec)

    x_ref = dataset[id_img][0].unsqueeze(0).to(device)
    physics = get_physics(shape, pb, device)
    y = physics(x_ref)  # A(x) + noise

    return x_ref, y


def main_exp(x_ref, y, pb, flag_ml_pnp, device):
    flag_pnp = not flag_ml_pnp
    denoiser = DRUNet(in_channels=3, out_channels=3, device=device, pretrained="download")
    denoiser = EquivariantDenoiser(denoiser)
    prior = PnP(denoiser=denoiser)
    data_fidelity = L2()

    # gridsearch values
    if pb == "inpainting":
        sigma_denoiser = 0.0801
        stepsize = 1.0
    elif pb == "demosaicing":
        sigma_denoiser = 0.0801
        stepsize = 1.17
    elif pb == "blur":
        sigma_denoiser = 0.0701
        stepsize = 1.0
    else:
        raise ValueError("pb has an unsupported value, it should be 'inpainting' or 'demosaicing' or 'blur'.")

    # hyper paramètres
    nb_iter_fine = 30
    nb_iter_coarse = 3  # in ML step
    nb_iter_coarse_init = 5  # in ML init
    nb_v_cycle = 2  # parameter "p"
    nb_levels = 4  # parameter "J"

    params_algo = {
        'g_param': sigma_denoiser,
        'stepsize': stepsize,

        'cit': SincFilter(),
        'backtracking': False,
        'ml_init': True,
        'n_levels': nb_levels,
        'iml_max_iter': nb_v_cycle,
        'coarse_iterator': CPGDIteration,

        'scale_coherent_grad': True,
        'scale_coherent_grad_init': False,
    }

    iter_vec = [nb_iter_coarse] * nb_levels
    iter_vec[-1] = nb_iter_fine
    stepsize_vec = [stepsize] * nb_levels

    ml_dict = {
        'iters': iter_vec,
        'stepsize': stepsize_vec,
    }

    params_algo['it_index'] = list(range(0, nb_iter_fine))
    if params_algo['ml_init'] is True:
        # number of iterations in coarse levels
        iters_init = [nb_iter_coarse_init] * (nb_levels - 1)
        iters_init.append(0)  # does not iterate on finest level
        ml_dict['iters_init'] = iters_init
        params_algo['level_init'] = nb_levels
        params_algo['multilevel_indices'] = [list(range(1, nb_v_cycle+1))]
    else:
        params_algo['multilevel_indices'] = [list(range(0, nb_v_cycle))]
    params_algo['params_multilevel'] = [ml_dict]
    params_algo['level'] = nb_levels

    gen0 = torch.Generator()
    gen0.manual_seed(2025)

    s0 = y.shape
    physics = get_physics((s0[-3], s0[-2], s0[-1]), pb, device)

    if flag_ml_pnp is True:
        model_ml_pnp = optim_builder(
            iteration=MultiLevelIteration(fine_iteration=PGDIteration()),
            prior=prior,
            data_fidelity=data_fidelity,
            max_iter=nb_iter_fine,
            g_first=False,
            early_stop=True,
            crit_conv='residual',
            thres_conv=1e-6,
            verbose=True,
            params_algo=params_algo,
        )

        model_ml_pnp.eval()
        x_est, met = model_ml_pnp(y, physics, x_gt=x_ref, compute_metrics=True)
        r_psnr = met['psnr'][0][-1]
    else:
        model_pnp = optim_builder(
            iteration=PGDIteration(),
            prior=prior,
            data_fidelity=data_fidelity,
            max_iter=nb_iter_fine,
            g_first=False,
            early_stop=True,
            crit_conv='residual',
            thres_conv=1e-6,
            verbose=True,
            params_algo=params_algo,
        )

        model_pnp.eval()
        x_est, met = model_pnp(y, physics, x_gt=x_ref, compute_metrics=True)
        r_psnr = met['psnr'][0][-1]


def main():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    id_img = 59
    pb = 'inpainting'
    x_ref, y = get_img(pb, device, id_img)

    print("=== Test memory consumption : without multilevel ===")
    flag_ml_pnp = False
    torch.cuda.reset_peak_memory_stats()
    main_exp(x_ref, y, pb, flag_ml_pnp, device)

    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
    print(f"Max GPU memory allocated: {max_memory:.2f} MB")

    # Reserved memory (includes cached but not used)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(f"Max GPU memory reserved: {max_reserved:.2f} MB")

    print("=== Test memory consumption : with multilevel ===")
    flag_ml_pnp = True
    torch.cuda.reset_peak_memory_stats()
    main_exp(x_ref, y, pb, flag_ml_pnp, device)

    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
    print(f"Max GPU memory allocated: {max_memory:.2f} MB")

    # Reserved memory (includes cached but not used)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(f"Max GPU memory reserved: {max_reserved:.2f} MB")


if __name__ == '__main__':
    main()