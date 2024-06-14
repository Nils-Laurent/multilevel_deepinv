from multilevel.info_transfer import BlackmannHarris
from tests.utils import standard_multilevel_param
from utils.get_hyper_param import inpainting_hyper_param, blur_hyper_param, tomography_hyper_param

def get_parameters_red(params_exp):
    params_algo, hp_red, hp_tv = get_param_algo_(params_exp)
    p_red = params_algo.copy()

    lambda_red = hp_red['lambda']
    g_param = hp_red['g_param']

    print("lambda_red:", lambda_red)

    lip_g = 160.0  # DRUnet lipschitz

    iters_fine = 200
    iters_coarse = 3
    iters_vec = [iters_coarse, iters_coarse, iters_coarse, iters_fine]
    p_red['iml_max_iter'] = 8

    p_red = standard_multilevel_param(p_red, it_vec=iters_vec)
    p_red['g_param'] = g_param
    p_red['lip_g'] = lip_g  # denoiser Lipschitz constant
    p_red['lambda'] = lambda_red
    p_red['step_coeff'] = 0.9  # no convex setting
    p_red['stepsize'] = p_red['step_coeff'] / (1.0 + lambda_red * lip_g)

    param_init = {'init_ml_x0': [80] * len(iters_vec)}
    return p_red, param_init

def get_parameters_tv(params_exp):
    params_algo, hp_red, hp_tv = get_param_algo_(params_exp)
    p_tv = params_algo.copy()
    lambda_tv = hp_tv['lambda']

    print("lambda_tv:", lambda_tv)

    iters_fine = 200
    iters_coarse = 3
    iters_vec = [iters_coarse, iters_coarse, iters_coarse, iters_fine]
    p_tv['iml_max_iter'] = 3

    p_tv = standard_multilevel_param(p_tv, it_vec=iters_vec)
    p_tv['lambda'] = lambda_tv
    p_tv['lip_g'] = 1.0  # denoiser Lipschitz constant
    p_tv['prox_crit'] = 1e-6
    p_tv['prox_max_it'] = 1000
    p_tv['params_multilevel'][0]['gamma_moreau'] = [1.1] * len(iters_vec)  # smoothing parameter
    p_tv['params_multilevel'][0]['gamma_moreau'][-1] = 1.0  # fine smoothing parameter
    p_tv['step_coeff'] = 1.9  # convex setting
    p_tv['stepsize'] = p_tv['step_coeff'] / (1.0 + lambda_tv)

    return p_tv

def get_param_algo_(params_exp):
    noise_pow = params_exp["noise_pow"]
    problem = params_exp['problem']

    print("def_noise:", noise_pow)

    if problem == 'inpainting':
        hp_red, hp_tv = inpainting_hyper_param(noise_pow)
    elif problem == 'blur':
        hp_red, hp_tv = blur_hyper_param(noise_pow)
    elif problem == 'tomography':
        hp_red, hp_tv = tomography_hyper_param(noise_pow)
    else:
        raise NotImplementedError("not implem")

    params_algo = {
        'cit': BlackmannHarris(),
        'scale_coherent_grad': True
    }

    return params_algo, hp_red, hp_tv