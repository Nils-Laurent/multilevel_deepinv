import copy
import torch

import deepinv
from deepinv.physics import GaussianNoise, Inpainting
from deepinv.utils import plot, plot_curves
from deepinv.models import DRUNet
from deepinv.optim import L2, optim_builder
from deepinv.optim.optim_iterators import GDIteration
from deepinv.optim.prior import ScorePrior

# multilevel imports
from multilevel.info_transfer import BlackmannHarris
from multilevel.iterator import MultiLevelIteration, MultiLevelParams
from multilevel.coarse_model import CoarseModel


def minimal_test():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    url = "https://culturezvous.com/wp-content/uploads/2017/10/chateau-azay-le-rideau.jpg?download=true"
    img_size = 256 if torch.cuda.is_available() else 64
    x_ref = deepinv.utils.load_url_image(url=url, img_size=img_size).to(device)

    noise_pow = 0.1
    inpainting = 0.8  # proportion of pixels to keep

    g = GaussianNoise(sigma=noise_pow)
    physics = Inpainting(x_ref.shape[1:], mask=inpainting, noise_model=g, device=device)
    y = physics(x_ref)

    lambda_red = 0.2 * noise_pow
    lip_d = 160.0  # DRUnet
    g_param = 0.05  # 0.05

    iters_fine = 400
    iters_vec = [5, 5, 5, iters_fine]
    if device == "cpu":
        iters_vec = [5, 5, iters_fine]

    levels = len(iters_vec)
    p_multilevel = MultiLevelParams({"iters": iters_vec})


    lambda_vec = [lambda_red / 4 ** i for i in range(0, levels)]
    lambda_vec.reverse()
    p_multilevel.params['lambda'] = lambda_vec

    step_coeff = 0.9
    stepsize_vec = [step_coeff / (l0 * lip_d + 1.0) for l0 in lambda_vec]
    p_multilevel.params['stepsize'] = stepsize_vec
    p_multilevel.params['verbose'] = [False] * levels
    p_multilevel.params['verbose'][levels - 1] = True

    params_red = {
        'cit': BlackmannHarris(),
        'level': levels,
        'params_multilevel': p_multilevel,
        'iml_max_iter': 5,
        'g_param': g_param,
        'scale_coherent_grad': True
    }

    data_fidelity = L2()
    denoiser = DRUNet(pretrained="download", train=False, device=device)
    prior = ScorePrior(denoiser) # Score prior divides by g_param**2
    iteration_level = GDIteration(has_cost=False)
    iteration = MultiLevelIteration(iteration_level)

    params_init = copy.deepcopy(params_red)
    params_init['params_multilevel'].params['iters'] = [80] * levels
    def init_ml_x0(x_, physics_):
        coarse_object = CoarseModel(prior, data_fidelity, physics_, params_init)
        x0_ = coarse_object.init_ml_x0({'est': [x_]}, x_, params_init)
        return {'est': [x0_], 'cost': None}
    f_init = init_ml_x0

    iters = params_red['params_multilevel'].params['iters'][-1]

    model = optim_builder(
        iteration=iteration,
        prior=prior,
        data_fidelity=data_fidelity,
        max_iter=iters,
        g_first=False,
        early_stop=True,
        crit_conv='residual',
        thres_conv=1e-6,
        verbose=True,
        params_algo=copy.deepcopy(params_red),
        custom_init=f_init
    )

    x_est, met = model(y, physics, x_gt=x_ref, compute_metrics=True)

    x0 = f_init(y, physics)['est'][0]
    plot([x_est, x0, x_ref, y], titles=["est", "x0", "ref", "y"])
    plot_curves(met)

if __name__ == "__main__":
    minimal_test()