import utils.ml_dataclass as dc
import utils.ml_dataclass_denoiser as dcn
import torch
import numpy
from deepinv.physics import GaussianNoise

from utils.parameters_global import ConfParam
from utils.parameters_utils import set_multilevel_init_params
from utils.run_alg import RunAlgorithm
from utils.utils import physics_from_exp, data_from_user_input

from multilevel_utils.custom_poisson_noise import CPoissonNoise, CPoissonLikelihood

def tune_grid_all(data_in, params_exp, device):
    params_exp['gridsearch'] = {}
    noise_pow = params_exp["noise_pow"]
    print("def_noise:", noise_pow)

    tensor_np = torch.tensor(float(noise_pow)).to(device)
    if isinstance(ConfParam().data_fidelity(), CPoissonLikelihood):
        bkg = ConfParam().data_fidelity().bkg
        gain = ConfParam().data_fidelity().gain
        g = CPoissonNoise(gain=gain)
    else:
        g = GaussianNoise(sigma=tensor_np)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    class_list = [
        dc.MFbMLGD,
        dc.MPnP,
        dc.MPnPInit,
        dc.MPnPML,
        dc.MPnPMLInit,
        dc.MPnPMoreau,
        dc.MPnPMoreauInit,
        dc.MPnPProx,
        dc.MPnPProxMLInit,
        #dcn.MPnPDnCNN,
        #dcn.MPnPMLDnCNNInit,
        dcn.MPnPSCUNet,
        dcn.MPnPMLSCUNetInit,
    ]

    #if not (params_exp['problem'] == 'mri'):
    #    class_list.append(dc.MPnPProx)
    #    class_list.append(dc.MPnPProxMLInit)

    print(f"==============================")
    print(f"GRIDSEARCH (device : {device}, pb : {params_exp['problem']})")
    print(f"==============================")
    res = {}
    with torch.no_grad():
        for m_class in class_list:
            print(f"=== device : {device}, m_class key : {m_class.key}, pb : {params_exp['problem']} ===")
            if 'cpu' in str(device):
                raise Exception("gridsearch should not run on CPU")

            def objective_fun(params_gs):
                params_exp['gridsearch'] = params_gs
                if 'stepsz_coeff' in params_gs.keys():
                    params_exp['stepsz_coeff'] = params_gs['stepsz_coeff']
                m_param = m_class.param_fn(params_exp, m_class)
                ra = RunAlgorithm(data, physics, params_exp, device=device, def_name="GS_"+m_class().key)
                if hasattr(m_class, 'use_init') and m_class.use_init is True:
                    set_multilevel_init_params(m_param)
                return ra.run_algorithm(m_class, m_param)

            res_data, res_keys = tune_algo(algo=objective_fun, alg_class=m_class, params_exp=params_exp)
            res[m_class.key] = {'axis': res_keys, 'tensors': res_data}

    return res

def tune_algo(algo, alg_class, params_exp):
    noise_pow = params_exp["noise_pow"]
    pb = params_exp["problem"]

    k_lambda = 'lambda'
    par_lambda = [[1E-5, 3.0], 13]
    k_sig = 'g_param'
    par_sig = [[0.0001, 0.50], 11]
    k_coeff = 'stepsz_coeff'
    par_coeff = [[0.0001, 3.00], 7]

    d_grid = {}
    recurse = 2
    if alg_class == dc.MPnPML or alg_class == dc.MPnPMLInit \
            or alg_class == dc.MPnP or alg_class == dc.MPnPInit:
        d_grid[k_sig] = par_sig
        d_grid[k_coeff] = par_coeff
    elif alg_class == dc.MPnPProxMLInit or alg_class == dc.MPnPProx:
        d_grid[k_sig] = par_sig
        d_grid[k_coeff] = [[1.0, 2.0], 5]
    elif alg_class == dcn.MPnPMLSCUNetInit or alg_class == dcn.MPnPSCUNet \
            or alg_class == dcn.MPnPMLDnCNNInit or alg_class == dcn.MPnPDnCNN:
        d_grid[k_coeff] = par_coeff
    elif alg_class == dc.MPnPMoreauInit or alg_class == dc.MPnPMoreau:
        d_grid[k_lambda] = [[1E-5, 3.0], 5]
        d_grid[k_sig] = [[0.01, 0.2], 5]
        d_grid[k_coeff] = [[0.0001, 3.00], 5]
    elif alg_class == dc.MRedMLInit:
        d_grid[k_lambda] = par_lambda
        d_grid[k_sig] = par_sig
    elif alg_class == dc.MFbMLGD:
        d_grid[k_lambda] = par_lambda
    else:
        raise ValueError("Invalid gridsearch class: {}".format(alg_class))

    return _tune(algo, d_grid, recurse)


def _tune(algo, d_grid, recurse, prec=None, log=False):
    TEST_FLAG = False

    recurse = recurse - 1
    if TEST_FLAG is True:
        recurse = 0
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
        params_gs = {}
        for j in range(len(sz)):
            q = id_map[j]
            kj = list(params_name)[j]
            params_gs[kj] = axis_vec[j][q].item()
            print(f"set {kj} to {params_gs[kj]}")

        if it >= 2 and TEST_FLAG is True:
            continue

        r_psnr = algo(params_gs)
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

    return _tune(algo, d_grid2, recurse, prec=prec)
