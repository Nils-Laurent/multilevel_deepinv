import utils.ml_dataclass as dc
import tests.parameters as P
import torch
import numpy
from deepinv.physics import GaussianNoise

from tests.test_alg import RunAlgorithm
from tests.utils import physics_from_exp, data_from_user_input


def tune_grid_all(data_in, params_exp, device):
    params_exp['gridsearch'] = True
    noise_pow = params_exp["noise_pow"]
    print("def_noise:", noise_pow)

    tensor_np = torch.tensor(float(noise_pow)).to(device)
    g = GaussianNoise(sigma=tensor_np)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    class_info = {
        dc.MPnPMLInit : {"coeff": 0.9},
        dc.MPnPMoreauInit : {"coeff": 0.9},  # parametre lambda
        dc.MFbMLGD : {"coeff": 1.9},
        #dc.MRedMLInit : {"coeff": 0.9},
    }
    if not (params_exp['problem'] == 'mri'):
        class_info[dc.MPnPProxMLInit] = {"coeff": 0.9}
    print(f"==============================")
    print(f"GRIDSEARCH (device : {device}, pb : {params_exp['problem']})")
    print(f"==============================")
    res = {}
    with torch.no_grad():
        for m_class in class_info.keys():
            print(f"=== device : {device}, m_class key : {m_class.key}, pb : {params_exp['problem']} ===")
            if 'cpu' in str(device):
                raise Exception("gridsearch should not run on CPU")

            m_param = m_class.param_fn(params_exp)
            m_param['step_coeff'] = class_info[m_class]["coeff"]
            def objective_fun(params):
                step_coeff = params['step_coeff']
                lip_g = params['lip_g']
                lambda_r = params['lambda']
                params['stepsize'] = step_coeff / (1.0 + lambda_r * lip_g)
                ra = RunAlgorithm(data, physics, params_exp, device=device, def_name="GS_"+m_class().key)
                if hasattr(m_class, 'use_init') and m_class.use_init is True:
                    p_init = P.get_multilevel_init_params(m_param)
                    p_init['stepsize'] = step_coeff / (1.0 + lambda_r * lip_g)
                    ra.set_init(p_init)
                return ra.run_algorithm(m_class, params)

            res_data, res_keys = tune_algo(m_param, algo=objective_fun, alg_class=m_class, params_exp=params_exp)
            res[m_class.key] = {'axis': res_keys, 'tensors': res_data}

    return res

def tune_algo(params_algo, algo, alg_class, params_exp):
    noise_pow = params_exp["noise_pow"]
    pb = params_exp["problem"]

    k_lambda = 'lambda'
    par_lambda = [[1E-5, 1.0], 11]
    k_sig = 'g_param'
    par_sig = [[0.0001, 0.25], 11]

    d_grid = {}
    recurse = 2
    if alg_class == dc.MPnPMLInit:
        d_grid[k_sig] = par_sig
    elif alg_class == dc.MPnPMoreauInit:
        d_grid[k_lambda] = par_lambda
        d_grid[k_sig] = par_sig
    elif alg_class == dc.MPnPProxMLInit:
        d_grid[k_sig] = par_sig
    elif alg_class == dc.MRedMLInit:
        if pb == "inpainting" and noise_pow >= 0.1:
            d_grid[k_lambda] = [[2.0, 12.0], 11]
            recurse = 3
        else:
            d_grid[k_lambda] = [[1E-5, 1.0], 11]
        d_grid[k_sig] = par_sig
    elif alg_class == dc.MFbMLGD:
        d_grid[k_lambda] = par_lambda
    else:
        raise ValueError("Invalid gridsearch class: {}".format(alg_class))

    return _tune(params_algo, algo, d_grid, recurse)


def _tune(params_algo, algo, d_grid, recurse, prec=None, log=False):
    TEST_FLAG = True

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
            #y = torch.logspace(a, b, steps=split, base=10.0)[1:-1]
            y = torch.logspace(a, b, steps=split, base=10.0)
            print(f"parameter range: key = {key_}")
            print(f"value = {y}")
        else:
            #y = torch.linspace(r_range[0], r_range[1], split)[1:-1]
            y = torch.linspace(r_range[0], r_range[1], split)
        axis_vec.append(y)
        sz.append(len(y))

    cost_map = torch.full(sz, torch.nan)

    nb_iter = len(list(numpy.ndindex(cost_map.shape)))
    it = 0
    for id_map in numpy.ndindex(cost_map.shape):
        it += 1
        print(f"--- Iteration {it} of {nb_iter} ---")
        for j in range(len(sz)):
            q = id_map[j]
            kj = list(params_name)[j]
            params_algo[kj] = axis_vec[j][q].item()
            print(f"set {kj} to {params_algo[kj]}")

        if it >= 2 and TEST_FLAG is True:
            continue

        r_psnr = algo(params_algo.copy())
        t_psnr = torch.tensor(r_psnr)
        cost_map[id_map] = t_psnr

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
        return prec, list(d_grid.keys())

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
