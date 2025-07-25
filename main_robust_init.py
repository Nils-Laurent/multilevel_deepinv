import sys

#if "/.fork" in sys.prefix:
sys.path.append('/projects/UDIP/nils_src/deepinv')

import deepinv
import torch
from scipy.io import savemat
from torchvision import transforms
from deepinv.models import EquivariantDenoiser, DRUNet
from deepinv.optim import optim_builder, PnP, L2
from deepinv.optim.optim_iterators import PGDIteration
from deepinv.physics import Inpainting, GaussianNoise
from deepinv.utils.demo import load_dataset

from multilevel.coarse_pgd import CPGDIteration
from multilevel.info_transfer import SincFilter
from multilevel.iterator import MultiLevelIteration
from utils.paths import dataset_path

def get_physics(shape, device):
    noise_level = torch.tensor(0.1)
    noise_model = GaussianNoise(sigma=noise_level)
    physics = Inpainting(shape, mask=0.5, noise_model=noise_model, device=device)
    return physics


def get_img(device):
    shape = (3, 1024, 1024)  # rgb 1024x1024 pixels
    size = (1024, 1024)  # rgb 1024x1024 pixels

    tr_vec = transforms.Compose([transforms.CenterCrop(size), transforms.ToTensor(), ])
    dataset = load_dataset('LIU4K-v2', dataset_path(), transform=tr_vec)

    x_ref = dataset[66][0].unsqueeze(0).to(device)
    physics = get_physics(shape, device)
    y = physics(x_ref)  # A(x) + noise

    return x_ref, y


def main_exp(x_ref, y, N_rng, N_iter, device):
    denoiser = DRUNet(in_channels=3, out_channels=3, device=device, pretrained="download")
    denoiser = EquivariantDenoiser(denoiser)
    prior = PnP(denoiser=denoiser)
    data_fidelity = L2()

    # gridsearch values
    sigma_denoiser = 0.0801
    stepsize = 1.0

    # hyper paramètres
    nb_iter_fine = N_iter
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

    def custom_init_rand(y_in, physics_in):
        x0 = torch.rand(y_in.shape, generator=gen0).to(y_in.device)
        return {'est': [x0]}

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
        custom_init=custom_init_rand
    )

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
        custom_init=custom_init_rand
    )
    s0 = y.shape
    physics = get_physics((s0[-3], s0[-2], s0[-1]), device)

    vec_est_ml_pnp = []
    vec_psnr_ml_pnp = []
    vec_est_pnp = []
    vec_psnr_pnp = []
    if N_iter == 0:
        for i in range(0, N_rng):
            x0 = custom_init_rand(y, physics)['est'][0]
            vec_est_ml_pnp.append(x0)
        vec_est_pnp = vec_est_ml_pnp

        vec_psnr_ml_pnp = [0] * N_rng
        vec_psnr_pnp = vec_psnr_ml_pnp
    else:
        for i in range(0, N_rng):
            print(f"i = {i}")
            model_ml_pnp.eval()
            x_est, met = model_ml_pnp(y, physics, x_gt=x_ref, compute_metrics=True)
            r_psnr = met['psnr'][0][-1]
            vec_est_ml_pnp.append(x_est)
            vec_psnr_ml_pnp.append(r_psnr.item())

            model_pnp.eval()
            x_est, met = model_pnp(y, physics, x_gt=x_ref, compute_metrics=True)
            r_psnr = met['psnr'][0][-1]
            vec_est_pnp.append(x_est)
            vec_psnr_pnp.append(r_psnr.item())

    stack_est = torch.stack(vec_est_ml_pnp, dim=1).squeeze()
    vec_std = torch.std(stack_est, dim=0)
    std_ml_pnp = torch.mean(vec_std).item()
    print(f"average_std ML PnP = {std_ml_pnp}")

    stack_est = torch.stack(vec_est_pnp, dim=1).squeeze()
    vec_std = torch.std(stack_est, dim=0)
    std_pnp = torch.mean(vec_std).item()
    print(f"average_std PnP = {std_pnp}")

    print(vec_psnr_pnp)
    print(vec_psnr_ml_pnp)
    return std_ml_pnp, vec_psnr_ml_pnp, std_pnp, vec_psnr_pnp


def main():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    x_ref, y = get_img(device)
    N_rng = 200
    for N_iter in [0, 1, 2, 3, 7, 20, 55, 200]:
        std_ml, psnr_ml, std, psnr = main_exp(x_ref, y, N_rng, N_iter, device)

        mat_data = {'std_ml_pnp': std_ml, 'vec_psnr_ml_pnp': psnr_ml,
                    'std_pnp': std, 'vec_psnr_pnp': psnr}

        out_f = f"init_robust_iter{N_iter}_rng{N_rng}.mat"
        savemat(out_f, mat_data)


if __name__ == "__main__":
    main()
