from functools import reduce

from matplotlib import pyplot

import math
import matplotlib
import torch
import numpy
from deepinv.physics import GaussianNoise

from multilevel.info_transfer import BlackmannHarris
from tests.test_alg import RunAlgorithm
from tests.utils import physics_from_exp, data_from_user_input, standard_multilevel_param


def tune_grid_all(data_in, params_exp, device):
    noise_pow = params_exp["noise_pow"]
    print("def_noise:", noise_pow)

    g = GaussianNoise(sigma=noise_pow)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    iters_fine = 200
    it_coarse = 3
    iters_vec = [it_coarse, it_coarse, it_coarse, iters_fine]

    params_algo = {
        'cit': BlackmannHarris(),
        'iml_max_iter': 8,
        'scale_coherent_grad': True
    }

    p_red = params_algo.copy()
    p_red = standard_multilevel_param(p_red, it_vec=iters_vec)
    p_red['step_coeff'] = 0.9  # no convex setting
    p_red['lip_g'] = 200  # denoiser Lipschitz constant

    param_init = {'init_ml_x0': [80] * len(iters_vec)}
    ra_red = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init)

    # parameters for tv
    p_tv = params_algo.copy()
    p_tv = standard_multilevel_param(p_tv, it_vec=iters_vec)
    p_tv['lip_g'] = 1.0  # denoiser Lipschitz constant
    p_tv['prox_crit'] = 1e-6
    p_tv['prox_max_it'] = 1000
    p_tv['params_multilevel'][0]['gamma_moreau'] = [1.1] * len(iters_vec)  # smoothing parameter
    p_tv['params_multilevel'][0]['gamma_moreau'][-1] = 1.0  # fine smoothing parameter
    p_tv['step_coeff'] = 1.9  # convex setting

    ra_tv = RunAlgorithm(data, physics, params_exp, device=device)

    # TUNE RED
    res_red, data_red, keys_red = tune_grid_red(p_red, ra_red.RED_GD, noise_pow)

    # TUNE TV
    res_tv, data_tv, keys_tv = tune_grid_tv(p_tv, ra_tv.TV_PGD, noise_pow)

    return {'res_tv': res_tv, 'data_tv': data_tv, 'keys_tv': keys_tv,
            'res_red': res_red, 'data_red': data_red, 'keys_red': keys_red}


def tune_grid_red(params_algo, algo, noise_pow):
    lambda_range = [0.01 * noise_pow, 1.0 * noise_pow]
    lambda_split = 9  # should be around 20
    sigma_range = [0.035, 0.2]
    sigma_split = 7  # should be around 15

    d_grid = {
        'lambda': [lambda_range, lambda_split],
        'g_param': [sigma_range, sigma_split],
    }

    recurse = 2
    return _tune(params_algo, algo, d_grid, recurse)


def tune_grid_tv(params_algo, algo, noise_pow):
    lambda_range = [0.01 * noise_pow, 3.0 * noise_pow]
    lambda_split = 15  # should be around 15

    d_grid = {
        'lambda': [lambda_range, lambda_split],
    }

    recurse = 2
    return _tune(params_algo, algo, d_grid, recurse)


def _tune(params_algo, algo, d_grid, recurse, prec=None):
    recurse = recurse - 1
    sz = []
    params_name = d_grid.keys()
    axis_vec = []
    for key_ in params_name:
        r_range = d_grid[key_][0]
        split = d_grid[key_][1]
        y = torch.linspace(r_range[0], r_range[1], split)[1:-1]
        axis_vec.append(y)
        sz.append(len(y))

    cost_map = torch.full(sz, torch.nan)

    nb_iter = len(list(numpy.ndindex(cost_map.shape)))
    it = 0
    for id_map in numpy.ndindex(cost_map.shape):
        it += 1
        print("-------------------------------------------------")
        print("Iteration {} of {}".format(it, nb_iter))
        for j in range(len(sz)):
            q = id_map[j]
            kj = list(params_name)[j]
            params_algo[kj] = axis_vec[j][q].item()
            print(f"set {kj} to {params_algo[kj]}")

        step_coeff = params_algo['step_coeff']
        lip_g = params_algo['lip_g']
        lambda_r = params_algo['lambda']
        params_algo['stepsize'] = step_coeff / (1.0 + lambda_r * lip_g)
        try:
            r = algo(params_algo.copy())
        except:
            print("Skip iteration: algorithm failed to run with current parameters")
            continue

        r_psnr = r['test_psnr']
        if not math.isnan(r_psnr):
            cost_map[id_map] = r_psnr
        print(f"iter {it} out of {nb_iter} (psnr {r['test_psnr']}, recurse {recurse})")

    # for tests only
    #cost_map = torch.rand(cost_map.shape)

    max_j = torch.argmax(cost_map.view(-1))
    max_i = torch.unravel_index(max_j, cost_map.shape)

    if prec is None:
        prec = [{'cost': cost_map, 'coord': axis_vec}]
    else:
        prec.append({'cost': cost_map, 'coord': axis_vec})

    if recurse == 0:
        res = {}
        for j in range(len(sz)):
            kj = list(params_name)[j]
            val_j = axis_vec[j][max_i[j]]
            res[kj] = val_j
        return res, prec, list(d_grid.keys())

    d_grid2 = d_grid.copy()
    for j in range(len(sz)):
        kj = list(params_name)[j]
        id_map = max_i[j]
        i_prec = max(0, id_map.item() - 1)
        i_next = min(sz[j] - 1, id_map.item() + 1)
        y_min = axis_vec[j][i_prec]
        y_max = axis_vec[j][i_next]
        # change range
        d_grid2[kj][0][0] = y_min.item()
        d_grid2[kj][0][1] = y_max.item()

    return _tune(params_algo, algo, d_grid2, recurse, prec=prec)


def tune_scatter_2d(d_tune, keys):
    v_min = numpy.min(list(map(lambda el: torch.min(el['cost']), d_tune)))
    v_max = numpy.max(list(map(lambda el: torch.max(el['cost']), d_tune)))

    s = 8.0
    for rec in range(len(d_tune)):
        coord = d_tune[rec]['coord']
        cost = d_tune[rec]['cost']

        x = []
        y = []
        z = []
        for id_xy in numpy.ndindex(cost.shape):
            x.append(coord[0][id_xy[0]])
            y.append(coord[1][id_xy[1]])
            z.append(cost[id_xy])
        pyplot.scatter(x, y, c=z, s=s, cmap='copper', vmin=v_min, vmax=v_max)
        s *= 0.7

    pyplot.xlabel(keys[0])
    pyplot.ylabel(keys[1])
    pyplot.colorbar()
    pyplot.show()




def tune_plot_1d(d_tune, keys):
    for rec in range(len(d_tune)):
        coord = d_tune[rec]['coord']
        cost = d_tune[rec]['cost']

        x = []
        y = []
        for id_xy in numpy.ndindex(cost.shape):
            x.append(coord[0][id_xy[0]])
            y.append(cost[id_xy])
        pyplot.plot(x, y)

    pyplot.xlabel(keys[0])
    pyplot.show()

