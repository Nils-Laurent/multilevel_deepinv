import deepinv
import torch
from deepinv.models import DRUNet
from deepinv.optim import PnP, optim_builder, L2
from deepinv.optim.optim_iterators import PGDIteration
from deepinv.physics import Demosaicing, GaussianNoise

from multilevel.coarse_pgd import CPGDIteration
from multilevel.info_transfer import SincFilter
from multilevel.iterator import MultiLevelIteration


def main_fn():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    # hyper paramètres
    lambda_pnp = 1.0
    zeta_denoiser = 0.05
    denoiser = DRUNet(in_channels=3, out_channels=3, device=device, pretrained="download")
    prior = PnP(denoiser=denoiser)
    nb_iter_fine = 200
    nb_iter_coarse = 3  # M_j où j < J
    nb_v_cycle = 2  # paramètre "p" dans le papier
    nb_levels = 4  # paramètre "J"
    stepsize = 1.0

    params_algo = {
        'lambda': lambda_pnp,
        'g_param': zeta_denoiser,
        'stepsize': stepsize,
        'lip_g': 1.0,

        'cit': SincFilter(),
        'prior': prior,
        'backtracking': False,
        'n_levels': nb_levels,
        'iml_max_iter': nb_v_cycle,
        'coarse_iterator': CPGDIteration, # Phi_j forward-backward

        'scale_coherent_grad': True,
        'scale_coherent_grad_init': False,
    }

    iter_vec = [nb_iter_coarse] * nb_levels
    iter_vec[-1] = nb_iter_fine
    stepsize_vec = [stepsize] * nb_levels

    lambda_vec = [lambda_pnp / 4 ** k for k in range(nb_levels-1, -1, -1)]
    ml_dict = {
        'iters': iter_vec,
        'lambda': lambda_vec,
        'stepsize': stepsize_vec,
    }

    params_algo['multilevel_step'] = [k < nb_v_cycle for k in range(0, nb_iter_fine)]
    params_algo['params_multilevel'] = [ml_dict]
    params_algo['level'] = nb_levels

    model = optim_builder(
        iteration=MultiLevelIteration(fine_iteration=PGDIteration()),
        prior=prior,
        data_fidelity=L2(),
        max_iter=nb_iter_fine,
        g_first=False,
        early_stop=True,
        crit_conv='residual',
        thres_conv=1e-6,
        verbose=True,
        params_algo=params_algo,
    )

    shape = (3, 1024, 1024)

    # remplacer x_ref par une image
    # attention : adapter la variable shape en conséquence
    x_ref = torch.randn(shape).to(device)

    noise_level = torch.tensor(0.1)
    noise_model = GaussianNoise(sigma=noise_level)
    physics = Demosaicing(shape, noise_model=noise_model, device=device)
    y = physics(x_ref)  # A(x) + noise
    model.eval()
    x_est, met = model(y, physics, x_gt=x_ref, compute_metrics=True)

if __name__ == '__main__':
    main_fn()