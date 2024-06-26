from deepinv.optim.prior import ScorePrior, RED, PnP, TVPrior
from deepinv.models import DRUNet

from multilevel.coarse_gradient_descent import CGDIteration
from multilevel.coarse_pgd import CPGDIteration
from multilevel.info_transfer import BlackmannHarris
from utils.get_hyper_param import inpainting_hyper_param, blur_hyper_param, tomography_hyper_param


def get_parameters_pnp(params_exp):
    params_algo, hp_red, hp_tv = get_param_algo_(params_exp)
    p_pnp = params_algo.copy()

    iters_fine = 200
    iters_coarse = 3
    iters_vec = [iters_coarse, iters_coarse, iters_coarse, iters_fine]
    p_pnp['iml_max_iter'] = 8

    p_pnp = standard_multilevel_param(p_pnp, it_vec=iters_vec)
    p_pnp['coarse_iterator'] = CPGDIteration
    p_pnp['lip_g'] = prior_lipschitz(PnP, p_pnp, DRUNet)

    # todo : gridsearch lambda, g_param
    p_pnp['g_param'] = 0.05
    lambda_pnp = 1.0

    p_pnp['lambda'] = lambda_pnp
    p_pnp['step_coeff'] = 0.9  # no convex setting
    p_pnp['stepsize'] = p_pnp['step_coeff'] / (1.0 + lambda_pnp * p_pnp['lip_g'])

    return p_pnp


def get_parameters_red(params_exp):
    params_algo, hp_red, hp_tv = get_param_algo_(params_exp)
    p_red = params_algo.copy()

    p_red['g_param'] = hp_red['g_param']
    lambda_red = hp_red['lambda']

    print("lambda_red:", lambda_red)

    iters_fine = 200
    iters_coarse = 3
    iters_vec = [iters_coarse, iters_coarse, iters_coarse, iters_fine]
    p_red['iml_max_iter'] = 8

    p_red = standard_multilevel_param(p_red, it_vec=iters_vec)
    p_red['lip_g'] = prior_lipschitz(RED, p_red, DRUNet)
    p_red['lambda'] = lambda_red
    p_red['step_coeff'] = 0.9  # no convex setting
    p_red['stepsize'] = p_red['step_coeff'] / (1.0 + lambda_red * p_red['lip_g'])

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
    p_tv['lip_g'] = prior_lipschitz(TVPrior, p_tv)
    p_tv['prox_crit'] = 1e-6
    p_tv['prox_max_it'] = 1000
    p_tv['params_multilevel'][0]['gamma_moreau'] = [1.1] * len(iters_vec)  # smoothing parameter
    p_tv['params_multilevel'][0]['gamma_moreau'][-1] = 1.0  # fine smoothing parameter
    p_tv['step_coeff'] = 1.9  # convex setting
    p_tv['stepsize'] = p_tv['step_coeff'] / (1.0 + lambda_tv)

    return p_tv

def standard_multilevel_param(params, it_vec):
    levels = len(it_vec)
    ml_dict = {"iters": it_vec}
    params['params_multilevel'] = [ml_dict]
    params['level'] = levels
    params['n_levels'] = levels
    params['coarse_iterator'] = CGDIteration

    iml_max_iter = params['iml_max_iter']
    params['multilevel_step'] = [k < iml_max_iter for k in range(0, it_vec[-1])]

    return params


def single_level_params(params_ml):
    params = params_ml.copy()
    params['n_levels'] = 1
    params['level'] = 1
    params['iters'] = params_ml['params_multilevel'][0]['iters'][-1]

    return params


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


def red_drunet_lipschitz():
    # this is for f(x) = (DRUNet_{sigma} - Identity)(x)
    return 1.6


def prior_lipschitz(prior, param, denoiser=None):
    if prior in [PnP, ScorePrior, RED]:
        if not (denoiser is DRUNet):
            raise ValueError("Unsupported denoiser type, expected DRUNet")

    if prior is PnP:
        # this is for f(x) = DRUNet_{sigma}(x)
        return 1.6
    if prior is ScorePrior:
        return red_drunet_lipschitz() / (param['g_param']**2)
    if prior is RED:
        return red_drunet_lipschitz()
    if prior is TVPrior:
        return 1.0

    raise ValueError("Unsupported prior")
