from deepinv.optim.prior import ScorePrior, RED, PnP, TVPrior
from deepinv.models import DRUNet, GSDRUNet

from multilevel.coarse_gradient_descent import CGDIteration
from multilevel.coarse_pgd import CPGDIteration
from multilevel.info_transfer import BlackmannHarris
from utils.get_hyper_param import inpainting_hyper_param, blur_hyper_param, tomography_hyper_param


def _finalize_params(params, lambda_vec, stepsize_vec, gamma_vec=None):
    params['params_multilevel'][0]['lambda'] = lambda_vec
    params['lambda'] = lambda_vec[-1]
    params['params_multilevel'][0]['stepsize'] = stepsize_vec
    params['stepsize'] = stepsize_vec[-1]
    if not (gamma_vec is None):
        params['params_multilevel'][0]['gamma_moreau'] = gamma_vec
        params['gamma_moreau'] = gamma_vec[-1]

    return params

def get_parameters_pnp(params_exp):
    params_algo, hp_red, hp_pnp, hp_tv = get_param_algo_(params_exp)
    p_pnp = params_algo.copy()
    p_pnp['scale_coherent_grad'] = False

    p_pnp['g_param'] = hp_pnp['g_param']
    lambda_pnp = hp_pnp['lambda']
    print("lambda_pnp:", lambda_pnp)

    iters_fine = 200
    iters_coarse = 3
    iters_vec = [iters_coarse, iters_coarse, iters_coarse, iters_fine]
    p_pnp['iml_max_iter'] = 8

    p_pnp = standard_multilevel_param(p_pnp, it_vec=iters_vec, lambda_fine=lambda_pnp)
    p_pnp['coarse_iterator'] = CPGDIteration
    p_pnp['lip_g'] = prior_lipschitz(PnP, p_pnp, DRUNet)

    lambda_vec = p_pnp['params_multilevel'][0]['lambda']
    stepsize_vec = [0.9/(l + p_pnp['lip_g']) for l in lambda_vec]

    p_pnp = _finalize_params(p_pnp, lambda_vec, stepsize_vec)

    #param_init = p_pnp.copy()
    #param_init['init_ml_x0'] = [80] * len(iters_vec)
    param_init = {}
    return p_pnp, param_init

def get_parameters_pnp_prox(params_exp):
    params_algo, hp_red, hp_pnp, hp_tv = get_param_algo_(params_exp)
    p_pnp = params_algo.copy()

    p_pnp['g_param'] = hp_pnp['g_param']
    #lambda_pnp = hp_pnp['lambda']
    lambda_pnp = 2.0/3.0
    print("lambda_pnp:", lambda_pnp)

    iters_fine = 200
    iters_coarse = 3
    #iters_vec = [iters_coarse, iters_coarse, iters_coarse, iters_fine]
    iters_vec = [iters_coarse, iters_fine]
    p_pnp['iml_max_iter'] = 8

    p_pnp = standard_multilevel_param(p_pnp, it_vec=iters_vec, lambda_fine=lambda_pnp)
    p_pnp['coarse_iterator'] = CPGDIteration
    p_pnp['lip_g'] = prior_lipschitz(PnP, p_pnp, GSDRUNet)
    p_pnp['coarse_prior'] = False

    # CANNOT CHOOSE STEPSIZE : see S. Hurault Thesis, Theorem 19.
    lambda_vec = [lambda_pnp]  * len(iters_vec)
    stepsize_vec = [1.0/l for l in lambda_vec]
    stepsize_vec[0:-1] = [2.0] * (len(iters_vec) - 1)

    p_pnp = _finalize_params(p_pnp, lambda_vec, stepsize_vec)
    #param_init = p_pnp.copy()
    #param_init['init_ml_x0'] = [80] * len(iters_vec)
    param_init = {}
    return p_pnp, param_init


def get_parameters_red(params_exp):
    params_algo, hp_red, hp_pnp, hp_tv = get_param_algo_(params_exp)
    p_red = params_algo.copy()

    p_red['g_param'] = hp_red['g_param']
    lambda_red = hp_red['lambda']
    print("lambda_red:", lambda_red)

    iters_fine = 200
    iters_coarse = 3
    iters_vec = [iters_coarse, iters_coarse, iters_coarse, iters_fine]
    p_red['iml_max_iter'] = 8

    p_red = standard_multilevel_param(p_red, it_vec=iters_vec, lambda_fine=lambda_red)
    p_red['lip_g'] = prior_lipschitz(RED, p_red, DRUNet)
    lambda_vec = p_red['params_multilevel'][0]['lambda']
    step_coeff = 0.9  # non-convex setting
    lf = 1.0  # data-fidelity Lipschitz cst
    stepsize_vec = [step_coeff / (lf + l * p_red['lip_g']) for l in lambda_vec]
    stepsize_vec[-1] = step_coeff / (lf + lambda_vec[-1] * p_red['lip_g'])
    p_red = _finalize_params(p_red, lambda_vec=lambda_vec, stepsize_vec=stepsize_vec)
    #p_red['params_multilevel'][0]['stepsize'] = stepsize_vec  # smoothing parameter
    #p_red['stepsize'] = p_red['step_coeff'] / (1.0 + lambda_red * p_red['lip_g'])
    #p_red['lambda'] = lambda_red

    param_init = p_red.copy()
    param_init['init_ml_x0'] = [80] * len(iters_vec)
    return p_red, param_init

def get_parameters_tv(params_exp):
    # We assume regularization gradient is 1-Lipschitz
    params_algo, hp_red, hp_pnp, hp_tv = get_param_algo_(params_exp)
    p_tv = params_algo.copy()

    lambda_tv = hp_tv['lambda']
    print("lambda_tv:", lambda_tv)

    iters_fine = 200
    iters_coarse = 5
    iters_vec = [iters_coarse, iters_coarse, iters_coarse, iters_fine]
    iters_vec = [iters_coarse, iters_fine]
    p_tv['iml_max_iter'] = 3

    p_tv = standard_multilevel_param(p_tv, it_vec=iters_vec, lambda_fine=lambda_tv)
    p_tv['lip_g'] = prior_lipschitz(TVPrior, p_tv)
    p_tv['prox_crit'] = 1e-6
    p_tv['prox_max_it'] = 1000
    gamma_vec = [1.1] * len(iters_vec)
    gamma_vec[-1] = 1.0
    lambda_vec = p_tv['params_multilevel'][0]['lambda']
    lf = 1.0  # data-fidelity gradient lipschitz cst
    step_coeff = 1.9  # convex setting
    stepsize_vec = [step_coeff / (lf + 1.0/gamma) for gamma in gamma_vec]
    stepsize_vec[-1] = step_coeff / (lf + lambda_vec[-1])
    p_tv = _finalize_params(p_tv, lambda_vec, stepsize_vec, gamma_vec)
    #p_tv['params_multilevel'][0]['stepsize'] = stepsize_vec
    #p_tv['params_multilevel'][0]['gamma_moreau'] = gamma_vec  # smoothing parameter
    #p_tv['step_coeff'] = 1.9  # convex setting
    #p_tv['stepsize'] = p_tv['step_coeff'] / (1.0 + lambda_tv)
    #p_tv['lambda'] = lambda_tv

    return p_tv

def get_parameters_tv_coarse_pgd(params_exp):
    p_tv = get_parameters_tv(params_exp)
    p_tv['coarse_iterator'] = CPGDIteration
    return p_tv

def standard_multilevel_param(params, it_vec, lambda_fine):
    levels = len(it_vec)
    ml_dict = {"iters": it_vec}
    lambda_vec = [lambda_fine / 4 ** k for k in range(len(it_vec)-1, -1, -1)]
    ml_dict['lambda'] = lambda_vec  # smoothing parameter
    params['params_multilevel'] = [ml_dict]
    params['level'] = levels
    params['n_levels'] = levels
    params['coarse_iterator'] = CGDIteration
    params['coarse_prior'] = True

    iml_max_iter = params['iml_max_iter']
    params['multilevel_step'] = [k < iml_max_iter for k in range(0, it_vec[-1])]

    return params


def single_level_params(params_ml):
    params = params_ml.copy()
    params['n_levels'] = 1
    params['level'] = 1
    params['iters'] = params_ml['params_multilevel'][0]['iters'][-1]
    params['lambda'] = params_ml['params_multilevel'][0]['lambda'][-1]
    params['stepsize'] = params_ml['params_multilevel'][0]['stepsize'][-1]

    return params


def get_param_algo_(params_exp):
    noise_pow = params_exp["noise_pow"]
    problem = params_exp['problem']

    print("def_noise:", noise_pow)

    if problem == 'inpainting' or problem == 'demosaicing':
        hp_red, hp_pnp, hp_tv = inpainting_hyper_param(noise_pow)
    elif problem == 'blur':
        hp_red, hp_pnp, hp_tv = blur_hyper_param(noise_pow)
    elif problem == 'tomography':
        hp_red, hp_pnp, hp_tv = tomography_hyper_param(noise_pow)
    else:
        raise NotImplementedError("not implem")

    params_algo = {
        'cit': BlackmannHarris(),
        'scale_coherent_grad': True
    }

    return params_algo, hp_red, hp_pnp, hp_tv


def red_drunet_lipschitz():
    # this is for f(x) = (DRUNet_{sigma} - Identity)(x)
    return 1.6


def prior_lipschitz(prior, param, denoiser=None):
    if prior in [PnP, ScorePrior, RED]:
        if denoiser is GSDRUNet:
            return 1.0
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
