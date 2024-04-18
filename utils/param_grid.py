import torch
import numpy
from deepinv.physics import GaussianNoise

from multilevel.info_transfer import BlackmannHarris
from tests.test_alg import RunAlgorithm
from tests.utils import physics_from_exp, data_from_user_input, standard_multilevel_param


def tune_grid_all(data_in, params_exp, device, max_lv):
    noise_pow = params_exp["noise_pow"]
    print("def_noise:", noise_pow)

    g = GaussianNoise(sigma=noise_pow)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    iters_fine = 200
    lc = 3
    iters_vec = [lc, lc, lc, iters_fine]

    params_algo = {
        'cit': BlackmannHarris(),
        'iml_max_iter': 8,
        'scale_coherent_grad': True
    }

    p_red = params_algo.copy()
    p_red = standard_multilevel_param(p_red, it_vec=iters_vec)
    p_red['step_coeff'] = 0.9  # no convex setting

    param_init = {'init_ml_x0': [80] * len(iters_vec)}
    ra = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init)
    algo = ra.RED_GD

    # TUNE RED
    tune_grid(p_red, algo)

    # parameters for tv
    p_tv = params_algo.copy()
    p_tv = standard_multilevel_param(p_tv, it_vec=iters_vec)
    p_tv['lip_g'] = 1.0  # denoiser Lipschitz constant
    p_tv['prox_crit'] = 1e-6
    p_tv['prox_max_it'] = 1000
    p_tv['params_multilevel'][0]['gamma_moreau'] = [1.1] * len(iters_vec)  # smoothing parameter
    p_tv['params_multilevel'][0]['gamma_moreau'][-1] = 1.0  # fine smoothing parameter
    p_tv['step_coeff'] = 1.9  # convex setting

    ra = RunAlgorithm(data, physics, params_exp, device=device)
    algo = ra.TV_PGD

    # TUNE TV
    pop_vec = ['lip_g']
    tune_grid(p_tv, algo, pop=pop_vec)


def tune_grid(params_algo, algo, pop=None):
    lambda_range = [0.01, 10.0]
    lambda_split = 10
    sigma_range = [0.001, 0.3]
    sigma_split = 5

    d_grid = {
        'lambda': [lambda_range, lambda_split],
        'lip_g': [sigma_range, sigma_split],
    }

    if pop is not None:
        for key_ in pop:
            d_grid.pop(key_)

    recurse = 2
    _tune(params_algo, algo, d_grid, recurse)


def _tune(params_algo, algo, d_grid, recurse):
    recurse = recurse - 1
    sz = []
    params_name = d_grid.keys()
    y_vec = []
    for key_ in params_name:
        r_range = d_grid[key_][0]
        split = d_grid[key_][1]
        sz.append(split)
        y = torch.linspace(r_range[0], r_range[1], split)
        y_vec.append(y)

    g = torch.full(sz, -torch.inf)

    for i0 in numpy.ndindex(g.shape):
        params = {}
        for j in range(len(sz)):
            q = i0[j]
            kj = list(params_name)[j]
            params_algo[kj] = y_vec[j][q]

        step_coeff = params_algo['step_coeff']
        lambda_r = params_algo['lambda']
        lip_g = params_algo['lip_g']
        params_algo['stepsize'] = step_coeff / (1.0 + lambda_r * lip_g)
        #algo(params_algo)
        #print(i0)

    g = torch.randn(sz)

    max_j = torch.argmax(g.view(-1))
    max_i = torch.unravel_index(max_j, g.shape)

    if recurse == 0:
        return max_i

    d_grid2 = d_grid.copy()
    for j in range(len(sz)):
        kj = list(params_name)[j]
        i0 = max_i[j]
        i_prec = max(0, i0.item() - 1)
        i_next = min(sz[j] - 1, i0.item() + 1)
        y_min = y_vec[j][i_prec]
        y_max = y_vec[j][i_next]
        # change range
        d_grid2[kj][0][0] = y_min.item()
        d_grid2[kj][0][1] = y_max.item()

    _tune(params_algo, algo, d_grid, recurse)

