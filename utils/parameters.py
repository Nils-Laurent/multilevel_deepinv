from deepinv.optim.prior import RED, PnP, TVPrior, Zero
from deepinv.models import DRUNet, GSDRUNet

from multilevel.coarse_gradient_descent import CGDIteration
from multilevel.coarse_pgd import CPGDIteration
from multilevel.prior import TVPrior as CustTV
from utils.parameters_global import ConfParam, FixedParams
from utils.parameters_utils import get_param_algo_, prior_lipschitz, _set_iter_vec, \
    standard_multilevel_param

def get_parameters_pnp_dncnn(params_exp):
    device = params_exp['device']
    import utils.ml_dataclass_denoiser as dcn
    key_vec = [dcn.MPnPMLDnCNNMoreauInit.key]
    res = get_param_algo_(params_exp, key_vec)
    g_param = 0  # NOT USED (DnCNN is a blind model)
    # in case the smooth approx. of TV is used in coarse scales
    lambda_pnp = res[dcn.MPnPMLDnCNNMoreauInit.key]['lambda']
    if 'stepsz_coeff' in params_exp.keys():
        coeff = params_exp['stepsz_coeff']
    else:
        coeff = None
    return parameters_pnp_common(ConfParam().get_dncnn(device), g_param, lambda_pnp, coeff)

def get_parameters_pnp_scunet(params_exp):
    device = params_exp['device']
    import utils.ml_dataclass_denoiser as dcn
    key_vec = [dcn.MPnPMLSCUNetMoreauInit.key]
    res = get_param_algo_(params_exp, key_vec)
    g_param = 0  # NOT USED (SCUNet is a blind model)
    # in case the smooth approx. of TV is used in coarse scales
    lambda_pnp = res[dcn.MPnPMLSCUNetMoreauInit.key]['lambda']
    if 'stepsz_coeff' in params_exp.keys():
        coeff = params_exp['stepsz_coeff']
    else:
        coeff = None
    return parameters_pnp_common(ConfParam().get_scunet(device), g_param, lambda_pnp, coeff)

def get_parameters_pnp_drunet(params_exp):
    device = params_exp['device']
    import utils.ml_dataclass as dcl
    key_vec = [dcl.MPnPMLInit.key, dcl.MPnPMoreauInit.key]
    res = get_param_algo_(params_exp, key_vec)
    g_param = res[dcl.MPnPMLInit.key]['g_param']
    # in case the smooth approx. of TV is used in coarse scales
    lambda_pnp = res[dcl.MPnPMoreauInit.key]['lambda']
    if 'stepsz_coeff' in params_exp.keys():
        coeff = params_exp['stepsz_coeff']
    else:
        coeff = None
    return parameters_pnp_common(ConfParam().get_drunet(device), g_param, lambda_pnp, coeff)

def parameters_pnp_common(denoiser, g_param, lambda_pnp, step_size_coeff=None):
    p_pnp = ConfParam().default_param()
    p_pnp['g_param'] = g_param

    p_pnp['prior'] = PnP(denoiser=denoiser)
    p_pnp['prior'].eval()

    iters_fine = ConfParam().iters_fine
    iters_coarse = ConfParam().iter_coarse_pnp_map
    iters_vec = _set_iter_vec(iters_coarse, iters_fine, ConfParam().levels)
    p_pnp['iml_max_iter'] = ConfParam().iml_max_iter

    p_pnp = standard_multilevel_param(p_pnp, it_vec=iters_vec, lambda_fine=lambda_pnp)
    p_pnp['coarse_iterator'] = CPGDIteration
    p_pnp['lip_g'] = prior_lipschitz(PnP, p_pnp, DRUNet)

    lf = ConfParam().data_fidelity_lipschitz

    if step_size_coeff is None:
        step_size_coeff = 0.9
    if FixedParams().stepsize_coeff is not None:
        step_size_coeff = FixedParams().get_stepsize_coeff()
    print("stepsize =", step_size_coeff/lf)
    stepsize_vec = [step_size_coeff/lf] * ConfParam().levels # PGD : only depends on lipschitz of data-fidelity
    p_pnp['params_multilevel'][0]['stepsize'] = stepsize_vec

    #p_pnp = _finalize_params(p_pnp, lambda_vec, stepsize_vec)
    return p_pnp

def get_parameters_pnp_non_exp(params_exp):
    import utils.ml_dataclass as dcl
    key_vec = [dcl.MPnPMLInit.key]
    res = get_param_algo_(params_exp, key_vec)
    p_pnp = ConfParam().default_param()

    p_pnp['g_param'] = res[dcl.MPnPMLInit.key]['g_param']
    lambda_pnp = res[dcl.MPnPMLInit.key]['lambda']
    print("lambda NE :", lambda_pnp)
    print("g_param NE :", p_pnp['g_param'])

    device = params_exp['device']
    denoiser = ConfParam().get_dncnn_nonexp(device)
    p_pnp['prior'] = PnP(denoiser=denoiser)
    p_pnp['prior'].eval()

    iters_fine = ConfParam().iters_fine
    iters_coarse = ConfParam().iter_coarse_pnp_map
    iters_vec = _set_iter_vec(iters_coarse, iters_fine, ConfParam().levels)
    p_pnp['iml_max_iter'] = ConfParam().iml_max_iter

    p_pnp = standard_multilevel_param(p_pnp, it_vec=iters_vec, lambda_fine=lambda_pnp)
    p_pnp['coarse_iterator'] = CPGDIteration
    p_pnp['lip_g'] = 1.0

    #lambda_vec = p_pnp['params_multilevel'][0]['lambda']
    lf = ConfParam().data_fidelity_lipschitz

    if 'stepsz_coeff' in params_exp.keys():
        step_size_coeff = params_exp['stepsz_coeff']
    else:
        step_size_coeff = 0.9
    if FixedParams().stepsize_coeff is not None:
        step_size_coeff = FixedParams().get_stepsize_coeff()
    stepsize_vec = [step_size_coeff/lf] * ConfParam().levels # PGD : only depends on lipschitz of data-fidelity
    stepsize_vec[-1] = stepsize_vec/lf  # PGD : only depends on lipschitz of data-fidelity
    p_pnp['params_multilevel'][0]['stepsize'] = stepsize_vec

    return p_pnp

def get_parameters_pnp_prox(params_exp):
    import utils.ml_dataclass as dcl
    key_vec = [dcl.MPnPProxMLInit.key]
    res = get_param_algo_(params_exp, key_vec)
    p_pnp = ConfParam().default_param()

    p_pnp['g_param'] = res[dcl.MPnPProxMLInit.key]['g_param']
    lambda_pnp = 2.0 /3.0  # see further down for this choice
    print("lambda_pnp_prox:", lambda_pnp)
    print("g_param_pnp_prox:", p_pnp['g_param'])

    device = params_exp['device']
    denoiser = ConfParam().get_gsdrunet(device)
    p_pnp['prior'] = PnP(denoiser=denoiser)

    iters_fine = ConfParam().iters_fine
    iters_coarse = ConfParam().iter_coarse_pnp_pgd
    print("iters_coarse:", iters_coarse)
    iters_vec = _set_iter_vec(iters_coarse, iters_fine, ConfParam().levels)
    p_pnp['iml_max_iter'] = ConfParam().iml_max_iter

    p_pnp = standard_multilevel_param(p_pnp, it_vec=iters_vec, lambda_fine=lambda_pnp)
    p_pnp['coarse_iterator'] = CPGDIteration
    p_pnp['lip_g'] = prior_lipschitz(PnP, p_pnp, GSDRUNet)
    p_pnp['backtracking'] = False

    # Cannot choose stepsize : see S. Hurault Thesis, Theorem 19.
    # lambda_pnp > 2 * data_fidelity_lipschitz / 3
    lambda_vec = [lambda_pnp]  * ConfParam().levels
    p_pnp['params_multilevel'][0]['lambda'] = lambda_vec
    if 'stepsz_coeff' in params_exp.keys():
        step_size_coeff = params_exp['stepsz_coeff']
    else:
        step_size_coeff = 1.0/lambda_pnp
    if FixedParams().stepsize_coeff is not None:
        step_size_coeff = FixedParams().get_stepsize_coeff()

    lf = ConfParam().data_fidelity_lipschitz
    stepsize_vec = [step_size_coeff/lf] * ConfParam().levels
    p_pnp['params_multilevel'][0]['stepsize'] = stepsize_vec

    return p_pnp


def get_parameters_red(params_exp):
    import utils.ml_dataclass as dcl
    key_vec = [dcl.MRedMLInit.key]
    res = get_param_algo_(params_exp, key_vec)
    p_red = ConfParam().default_param()

    p_red['g_param'] = res[dcl.MRedMLInit.key]['g_param']
    lambda_red = res[dcl.MRedMLInit.key]['lambda']
    print("lambda_red:", lambda_red)

    device = params_exp['device']
    denoiser = ConfParam().get_drunet(device)
    p_red['prior'] = RED(denoiser=denoiser)
    p_red['prior'].eval()

    iters_fine = ConfParam().iters_fine
    iters_coarse = ConfParam().iter_coarse_red
    iters_vec = _set_iter_vec(iters_coarse, iters_fine, ConfParam().levels)
    p_red['iml_max_iter'] = ConfParam().iml_max_iter

    p_red = standard_multilevel_param(p_red, it_vec=iters_vec, lambda_fine=lambda_red)
    p_red['lip_g'] = prior_lipschitz(RED, p_red, DRUNet)

    #p_red['params_multilevel'][0]['lambda'] = [lambda_red] * ConfParam().levels
    lambda_vec = p_red['params_multilevel'][0]['lambda']
    if 'stepsz_coeff' in params_exp.keys():
        step_size_coeff = params_exp['stepsz_coeff']
    else:
        step_size_coeff = 0.9  # non-convex setting
    if FixedParams().stepsize_coeff is not None:
        step_size_coeff = FixedParams().get_stepsize_coeff()
    lf = ConfParam().data_fidelity_lipschitz
    stepsize_vec = [step_size_coeff / (lf + l * p_red['lip_g']) for l in lambda_vec] # gradient descent
    p_red['params_multilevel'][0]['stepsize'] = stepsize_vec

    #p_red = _finalize_params(p_red, lambda_vec=lambda_vec, stepsize_vec=stepsize_vec)
    return p_red

def get_parameters_tv(params_exp):
    import utils.ml_dataclass as dcl
    key_vec = [dcl.MFbMLGD.key]
    # We assume regularization gradient is 1-Lipschitz
    res = get_param_algo_(params_exp, key_vec)
    p_tv = ConfParam().default_param()
    p_tv['scale_coherent_grad'] = True  # for FB TV we always use 1order coherence

    lambda_tv = res[dcl.MFbMLGD.key]['lambda']
    print("lambda_tv:", lambda_tv)

    crit = 1e-6
    tv_max_it = 1000
    p_tv['prior'] = CustTV(def_crit=crit, n_it_max=tv_max_it)
    p_tv['prior'].eval()

    iters_fine = ConfParam().iters_fine
    iters_coarse = ConfParam().iter_coarse_tv
    iters_vec = _set_iter_vec(iters_coarse, iters_fine, ConfParam().levels)
    p_tv['iml_max_iter'] = ConfParam().iml_max_iter

    p_tv = standard_multilevel_param(p_tv, it_vec=iters_vec, lambda_fine=lambda_tv)
    p_tv['scale_coherent_grad'] = True  # for FB TV we always use 1order coherence
    p_tv['lip_g'] = prior_lipschitz(TVPrior, p_tv)
    p_tv['prox_crit'] = crit
    p_tv['prox_max_it'] = tv_max_it
    gamma_vec = [1.1] * len(iters_vec)
    gamma_vec[-1] = 1.0
    p_tv['params_multilevel'][0]['gamma_moreau'] = gamma_vec
    lf = ConfParam().data_fidelity_lipschitz
    step_size_coeff = 1.9  # convex setting
    stepsize_vec = [step_size_coeff / (lf + 1.0/gamma) for gamma in gamma_vec]
    stepsize_vec[-1] = step_size_coeff / lf  # only depends on lipschitz of data-fidelity
    p_tv['params_multilevel'][0]['stepsize'] = stepsize_vec
    #p_tv = _finalize_params(p_tv, lambda_vec, stepsize_vec, gamma_vec)
    p_tv['scale_coherent_grad'] = True  # for FB TV we always use 1order coherence

    return p_tv

def get_parameters_dpir(params_exp):
    return {}

def get_parameters_tv_coarse_pgd(params_exp):
    p_tv = get_parameters_tv(params_exp)
    p_tv['coarse_iterator'] = CPGDIteration
    return p_tv


# ============== single level ==============
def single_level_params(params_ml):
    params = params_ml.copy()
    params['n_levels'] = 1
    params['level'] = 1
    params['iters'] = params_ml['params_multilevel'][0]['iters'][-1]
    params['lambda'] = params_ml['params_multilevel'][0]['lambda'][-1]
    params['stepsize'] = params_ml['params_multilevel'][0]['stepsize'][-1]

    return params

# ============== multilevel specific modifiers ==============
def set_ml_param_Moreau(params, params_exp):
    if isinstance(params['prior'], PnP):
        import utils.ml_dataclass as dcl
        key_vec = [dcl.MPnPMoreauInit.key]
        res = get_param_algo_(params_exp, key_vec)
        params['g_param'] = res[dcl.MPnPMoreauInit.key]['g_param']

    params['coarse_iterator'] = CGDIteration
    iters_vec = params['params_multilevel'][0]['iters']
    gamma_vec = [1.1] * len(iters_vec)
    gamma_vec[-1] = 1.0
    params['params_multilevel'][0]['gamma_moreau'] = gamma_vec
    params['gamma_moreau'] = gamma_vec[-1]

    params['coarse_prior'] = CustTV()

    params['params_multilevel'][0]['stepsize'] = [1.9 / (ConfParam().data_fidelity_lipschitz + 1/gamma_j) for gamma_j in gamma_vec]

    return params

def set_ml_param_student(params, params_exp):
    device = params_exp['device']
    if params_exp['problem'] == 'mri':
        denoiser = ConfParam().get_student1c(device)
    else:
        denoiser = ConfParam().get_student(device)

    prior_class = params['prior'].__class__
    params['coarse_prior'] = prior_class(denoiser=denoiser)

    return params

def set_ml_param_noreg(params, params_exp):
    assert False  # normally, it is not used anymore
    params['coherence_prior'] = params['prior']
    params['coarse_prior'] = Zero()
    return params

