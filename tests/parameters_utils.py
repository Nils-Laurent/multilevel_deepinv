from deepinv.models import GSDRUNet, DRUNet

from multilevel.coarse_gradient_descent import CGDIteration
from tests.parameters_global import ConfParam
from utils.get_hyper_param import inpainting_hyper_param, blur_hyper_param, mri_hyper_param, poisson_hyper_param, \
    demosaicing_hyper_param
from deepinv.optim.prior import ScorePrior, RED, PnP, TVPrior, Zero


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
    params['backtracking'] = True

    iml_max_iter = params['iml_max_iter']
    params['multilevel_step'] = [k < iml_max_iter for k in range(0, it_vec[-1])]

    #params['multilevel_step'][100 + -1] = True
    #params['multilevel_step'][100 + -2] = True

    return params

def set_new_nb_coarse(params):
    new_nb = 8
    l_iter = params['params_multilevel'][0]['iters'][0:-1]
    params['params_multilevel'][0]['iters'][0:-1] = [new_nb] * len(l_iter)
    print("iters_coarse:", new_nb)

def get_multilevel_init_params(params):
    iters_vec = params['params_multilevel'][0]['iters']
    param_init = params.copy()
    # does not iterate on finest level
    params['params_multilevel'][0]['iters_init'] = [ConfParam().coarse_iters_ini] * len(iters_vec)
    return param_init

def _set_iter_vec(it_coarse, it_fine, levels):
    vec = [it_coarse] * levels
    vec[-1] = it_fine
    return vec


def _finalize_params(params, lambda_vec, stepsize_vec, gamma_vec=None):
    params['params_multilevel'][0]['lambda'] = lambda_vec
    params['lambda'] = lambda_vec[-1]
    params['params_multilevel'][0]['stepsize'] = stepsize_vec
    params['stepsize'] = stepsize_vec[-1]
    if not (gamma_vec is None):
        params['params_multilevel'][0]['gamma_moreau'] = gamma_vec
        params['gamma_moreau'] = gamma_vec[-1]

    return params


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



def get_param_algo_(params_exp, key_vec):
    noise_pow = params_exp["noise_pow"]
    problem = params_exp['problem']

    res = {}

    print("def_noise:", noise_pow)
    if 'gridsearch' in params_exp.keys():
        for akey in key_vec:
            gsd = params_exp['gridsearch']
            res[akey] = {}
            if 'lambda' in gsd.keys():
                res[akey]['lambda'] = gsd['lambda']
            else:
                res[akey]['lambda'] = 0
            if 'g_param' in gsd.keys():
                res[akey]['g_param'] = gsd['g_param']
            else:
                res[akey]['g_param'] = 0
    elif problem == 'inpainting':
        for akey in key_vec:
            res[akey] = inpainting_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'demosaicing':
        for akey in key_vec:
            res[akey] = demosaicing_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'blur':
        for akey in key_vec:
            res[akey] = blur_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'mri':
        for akey in key_vec:
            res[akey] = mri_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'denoising':
        for akey in key_vec:
            res[akey] = poisson_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'tomography':
        pass
    else:
        raise NotImplementedError("not implem")

    return res


def red_drunet_lipschitz():
    # this is for f(x) = (DRUNet_{sigma} - Identity)(x)
    return 1.6
