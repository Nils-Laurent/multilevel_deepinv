import deepinv
import torch
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from deepinv.models import DRUNet, EquivariantDenoiser
from deepinv.optim import PnP, optim_builder, L2
from deepinv.optim.optim_iterators import PGDIteration
from deepinv.physics import Demosaicing, GaussianNoise, Inpainting

from multilevel.coarse_pgd import CPGDIteration
from multilevel.info_transfer import SincFilter
from multilevel.iterator import MultiLevelIteration


def main_fn():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    denoiser = DRUNet(in_channels=3, out_channels=3, device=device, pretrained="download")
    denoiser = EquivariantDenoiser(denoiser)
    prior = PnP(denoiser=denoiser)
    data_fidelity = L2()

    # hyper paramètres
    zeta_denoiser = 0.0601
    nb_iter_fine = 200
    nb_iter_coarse = 3  # M_j où j < J
    nb_iter_coarse_init = 5  # M_j for multilevel initialization
    nb_v_cycle = 2  # paramètre "p" dans le papier
    nb_levels = 4  # paramètre "J"
    stepsize = 0.9  # assuming data-fidelity is 1 Lipschitz

    params_algo = {
        'g_param': zeta_denoiser,
        'stepsize': stepsize,

        'cit': SincFilter(),
        'backtracking': False,
        'ml_init': True,
        'n_levels': nb_levels,
        'iml_max_iter': nb_v_cycle,
        'coarse_iterator': CPGDIteration, # Phi_j forward-backward

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

    model = optim_builder(
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


    url = "https://images.unsplash.com/photo-1733316006504-becb91c98310?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    shape = (3, 1024, 1024)  # rgb 1024x1024 pixels

    response = requests.get(url)
    print(response.status_code)
    img = Image.open(BytesIO(response.content))

    tr = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((shape[1], shape[2]))])
    x_ref = tr(img).to(device)

    noise_level = torch.tensor(0.1)
    noise_model = GaussianNoise(sigma=noise_level)
    physics = Inpainting(shape, mask=0.5, noise_model=noise_model, device=device)
    y = physics(x_ref)  # A(x) + noise
    model.eval()
    x_est, met = model(y, physics, x_gt=x_ref, compute_metrics=True)

    deepinv.utils.plot([x_ref, y, x_est])

if __name__ == '__main__':
    main_fn()