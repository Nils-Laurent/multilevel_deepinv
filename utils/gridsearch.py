from tests.parameters import get_parameters_red, get_parameters_tv

import math
import torch
import numpy
from deepinv.physics import GaussianNoise

from tests.test_alg import RunAlgorithm
from tests.utils import physics_from_exp, data_from_user_input


def tune_grid_all(data_in, params_exp, device):
    noise_pow = params_exp["noise_pow"]
    print("def_noise:", noise_pow)

    tensor_np = torch.tensor(noise_pow).to(device)
    g = GaussianNoise(sigma=tensor_np)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    # TUNE TV
    p_tv = get_parameters_tv(params_exp)
    ra_tv = RunAlgorithm(data, physics, params_exp, device=device)
    res_tv, data_tv, keys_tv = tune_grid_tv(p_tv, ra_tv.TV_PGD)

    # TUNE RED
    p_red, param_init = get_parameters_red(params_exp)
    ra_red = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init)
    res_red, data_red, keys_red = tune_grid_red(p_red, ra_red.RED_GD)

    return {'res_tv': res_tv, 'data_tv': data_tv, 'keys_tv': keys_tv,
            'res_red': res_red, 'data_red': data_red, 'keys_red': keys_red}


def tune_grid_red(params_algo, algo):
    lambda_range = [1E-6, 100.0]
    lambda_split = 11  # should be around 11
    sigma_range = [0.005, 0.51]
    sigma_split = 9  # should be around 9

    d_grid = {
        'lambda': [lambda_range, lambda_split],
        'g_param': [sigma_range, sigma_split],
    }

    recurse = 2
    return _tune(params_algo, algo, d_grid, recurse)


def tune_grid_tv(params_algo, algo):
    lambda_range = [1E-5, 300.0]
    lambda_split = 11  # should be around 11

    d_grid = {
        'lambda': [lambda_range, lambda_split],
    }

    recurse = 2
    return _tune(params_algo, algo, d_grid, recurse)


def _tune(params_algo, algo, d_grid, recurse, prec=None, log=True):
    TEST_FLAG = False

    recurse = recurse - 1
    sz = []
    params_name = d_grid.keys()
    axis_vec = []
    for key_ in params_name:
        r_range = d_grid[key_][0]
        split = d_grid[key_][1]
        if log is True:
            a = numpy.log10(r_range[0])
            b = numpy.log10(r_range[1])
            y = torch.logspace(a, b, steps=split, base=10.0)[1:-1]
            print(f"parameter range: key = {key_}")
            print(f"value = {y}")
        else:
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
        #try:
        if TEST_FLAG is True:
            continue

        r = algo(params_algo.copy())
        #except:
        #    print("Skip iteration: algorithm failed to run with current parameters")
        #    continue

        r_psnr = numpy.mean(r.final_values('psnr'))
        t_psnr = torch.tensor(r_psnr)
        cost_map[id_map] = t_psnr
        #if not math.isnan(r_psnr):
        #    cost_map[id_map] = t_psnr
        print(f"iter {it} out of {nb_iter} (psnr {r_psnr}, recurse {recurse})")

    # for tests only
    if TEST_FLAG is True:
        cost_map = torch.rand(cost_map.shape)

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
