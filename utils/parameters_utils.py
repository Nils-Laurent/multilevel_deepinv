from deepinv.models import GSDRUNet, DRUNet

from multilevel.coarse_gradient_descent import CGDIteration
from utils.parameters_global import ConfParam, FixedParams
from utils.get_hyper_param import inpainting_hyper_param, blur_hyper_param, mri_hyper_param, poisson_hyper_param, \
    demosaicing_hyper_param
from deepinv.optim.prior import ScorePrior, RED, PnP, TVPrior


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
    params['ml_init'] = False

    iml_max_iter = params['iml_max_iter']
    ml_indices = list(range(0, iml_max_iter))
    #ml_indices.append(100 + -1)
    #ml_indices.append(100 + -2)
    params['multilevel_indices'] = [ml_indices]
    params['it_index'] = list(range(0, ConfParam().iters_fine))

    return params

def set_new_nb_coarse(params):
    new_nb = 8
    l_iter = params['params_multilevel'][0]['iters'][0:-1]
    params['params_multilevel'][0]['iters'][0:-1] = [new_nb] * len(l_iter)
    print("iters_coarse:", new_nb)

def set_multilevel_init_params(params):
    iters_init = [ConfParam().coarse_iters_ini] * (ConfParam().levels - 1)
    iters_init.append(None)  # does not iterate on finest level
    params['params_multilevel'][0]['iters_init'] = iters_init
    params['level_init'] = ConfParam().levels
    params['ml_init'] = True

    # add one ml block since the 1st one is replaced by the init
    params['multilevel_indices'][0].append(params['iml_max_iter'])

    return params

def _set_iter_vec(it_coarse, it_fine, levels):
    vec = [it_coarse] * levels
    vec[-1] = it_fine
    return vec


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



def get_param_algo_(params_exp, method_key_vec):
    noise_pow = params_exp["noise_pow"]
    problem = params_exp['problem']

    res = {}

    print("def_noise:", noise_pow)
    if 'gridsearch' in params_exp.keys():
        for akey in method_key_vec:
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
        for akey in method_key_vec:
            res[akey] = inpainting_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'demosaicing':
        for akey in method_key_vec:
            res[akey] = demosaicing_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'blur':
        for akey in method_key_vec:
            res[akey] = blur_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'mri':
        for akey in method_key_vec:
            res[akey] = mri_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'denoising':
        for akey in method_key_vec:
            res[akey] = poisson_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'tomography':
        pass
    else:
        raise NotImplementedError("not implem")

    if FixedParams().g_param is not None:
        for akey in method_key_vec:
            res[akey]['g_param'] = FixedParams().get_g_param()

    return res


def red_drunet_lipschitz():
    # this is for f(x) = (DRUNet_{sigma} - Identity)(x)
    return 1.6

def use_init(m_class):
    return hasattr(m_class, 'use_init') and (m_class.use_init is True)
