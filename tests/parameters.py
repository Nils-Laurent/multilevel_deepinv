import os

import deepinv
from deepinv.optim.prior import ScorePrior, RED, PnP, TVPrior, Zero
from deepinv.models import DRUNet, GSDRUNet

from multilevel.approx_nn import Student
from multilevel.coarse_gradient_descent import CGDIteration
from multilevel.coarse_pgd import CPGDIteration
from multilevel.info_transfer import BlackmannHarris, CFir
from multilevel.prior import TVPrior as CTV
from utils.get_hyper_param import inpainting_hyper_param, blur_hyper_param
from utils.paths import checkpoint_path

from multilevel_utils.complex_denoiser import to_complex_denoiser


state_file_v3 = os.path.join(checkpoint_path(), 'student_v3_cs_c32_ic2_10L_525.pth.tar')
state_file_v4 = os.path.join(checkpoint_path(), 'student_v4_cs_c32_ic2_10L_weight2_599.pth.tar')


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ConfParam(metaclass=Singleton):
    win = None
    levels = 0
    iters_fine = 0
    iml_max_iter = 0
    use_complex_denoiser = False
    denoiser_in_channels = 3

    def get_drunet(self, device):
        net = DRUNet(in_channels=self.denoiser_in_channels, out_channels=self.denoiser_in_channels, pretrained="download", device=device)
        denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        if self.use_complex_denoiser is True:
            denoiser = to_complex_denoiser(denoiser, mode="separated")
        return denoiser

    def get_student(self, device):
        d = Student(layers=10, nc=32, cnext_ic=2, pretrained=state_file_v3).to(device)
        if ConfParam().use_complex_denoiser is True:
            d = to_complex_denoiser(d, mode="separated")
        return d

    def get_gsdrunet(self, device):
        denoiser = GSDRUNet(pretrained="download", device=device)
        if ConfParam().use_complex_denoiser is True:
            denoiser = to_complex_denoiser(denoiser, mode="separated")
        return denoiser


def _set_iter_vec(it_coarse, it_fine):
    vec = [it_coarse] * ConfParam().levels
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

def get_parameters_pnp(params_exp):
    params_algo, res = get_param_algo_(params_exp)
    p_pnp = params_algo.copy()
    p_pnp['scale_coherent_grad'] = False

    import utils.ml_dataclass as dcl
    p_pnp['g_param'] = res[dcl.MPnPML.key]['g_param']
    lambda_pnp = res[dcl.MPnPML.key]['lambda']
    print("lambda_pnp:", lambda_pnp)

    device = params_exp['device']
    denoiser = ConfParam().get_drunet(device)
    p_pnp['prior'] = PnP(denoiser=denoiser)

    iters_fine = ConfParam().iters_fine
    iters_coarse = 8
    if params_exp['noise_pow'] <= 0.05:  # on off system : computation time
        iters_coarse = 20
    iters_vec = _set_iter_vec(iters_coarse, iters_fine)
    #p_pnp['iml_max_iter'] = 8
    #p_pnp['iml_max_iter'] = 1
    p_pnp['iml_max_iter'] = ConfParam().iml_max_iter

    p_pnp = standard_multilevel_param(p_pnp, it_vec=iters_vec, lambda_fine=lambda_pnp)
    p_pnp['coarse_iterator'] = CPGDIteration
    p_pnp['lip_g'] = prior_lipschitz(PnP, p_pnp, DRUNet)

    lambda_vec = p_pnp['params_multilevel'][0]['lambda']
    stepsize_vec = [0.9/(l + p_pnp['lip_g']) for l in lambda_vec]

    p_pnp = _finalize_params(p_pnp, lambda_vec, stepsize_vec)
    return p_pnp

def get_parameters_pnp_prox(params_exp):
    params_algo, res = get_param_algo_(params_exp)
    p_pnp = params_algo.copy()

    import utils.ml_dataclass as dcl
    p_pnp['g_param'] = res[dcl.MPnPProxML.key]['g_param']
    #lambda_pnp = hp_pnp['lambda']
    lambda_pnp = 2.0/3.0
    print("lambda_pnp:", lambda_pnp)

    device = params_exp['device']
    denoiser = ConfParam().get_gsdrunet(device)
    p_pnp['prior'] = PnP(denoiser=denoiser)

    iters_fine = ConfParam().iters_fine
    iters_coarse = 8
    if params_exp['noise_pow'] <= 0.05:  # on off system : computation time
        iters_coarse = 20

    print("iters_coarse:", iters_coarse)
    #iters_coarse = 5
    iters_vec = _set_iter_vec(iters_coarse, iters_fine)
    #p_pnp['iml_max_iter'] = 1
    p_pnp['iml_max_iter'] = ConfParam().iml_max_iter

    p_pnp = standard_multilevel_param(p_pnp, it_vec=iters_vec, lambda_fine=lambda_pnp)
    p_pnp['coarse_iterator'] = CPGDIteration
    p_pnp['lip_g'] = prior_lipschitz(PnP, p_pnp, GSDRUNet)
    p_pnp['backtracking'] = False
    p_pnp['scale_coherent_grad'] = True

    # CANNOT CHOOSE STEPSIZE : see S. Hurault Thesis, Theorem 19.
    lambda_vec = [lambda_pnp]  * len(iters_vec)
    stepsize_vec = [1.0/l for l in lambda_vec]
    stepsize_vec[0:-1] = [2.0] * (len(iters_vec) - 1)

    p_pnp = _finalize_params(p_pnp, lambda_vec, stepsize_vec)
    return p_pnp


def get_parameters_red(params_exp):
    params_algo, res = get_param_algo_(params_exp)
    p_red = params_algo.copy()

    import utils.ml_dataclass as dcl
    p_red['g_param'] = res[dcl.MRedMLInit.key]['g_param']
    lambda_red = res[dcl.MRedMLInit.key]['lambda']
    print("lambda_red:", lambda_red)

    device = params_exp['device']
    denoiser = ConfParam().get_drunet(device)
    p_red['prior'] = RED(denoiser=denoiser)

    iters_fine = ConfParam().iters_fine
    #iters_coarse = 3
    iters_coarse = 8
    if params_exp['noise_pow'] <= 0.05:  # on off system : computation time
        iters_coarse = 20
    iters_vec = _set_iter_vec(iters_coarse, iters_fine)
    #p_red['iml_max_iter'] = 8
    #p_red['iml_max_iter'] = 1
    p_red['iml_max_iter'] = ConfParam().iml_max_iter

    p_red = standard_multilevel_param(p_red, it_vec=iters_vec, lambda_fine=lambda_red)
    p_red['lip_g'] = prior_lipschitz(RED, p_red, DRUNet)
    lambda_vec = p_red['params_multilevel'][0]['lambda']
    step_coeff = 0.9  # non-convex setting
    lf = 1.0  # data-fidelity Lipschitz cst
    stepsize_vec = [step_coeff / (lf + l * p_red['lip_g']) for l in lambda_vec]
    stepsize_vec[-1] = step_coeff / (lf + lambda_vec[-1] * p_red['lip_g'])
    p_red = _finalize_params(p_red, lambda_vec=lambda_vec, stepsize_vec=stepsize_vec)

    return p_red

def get_parameters_tv(params_exp):
    # We assume regularization gradient is 1-Lipschitz
    params_algo, res = get_param_algo_(params_exp)
    p_tv = params_algo.copy()

    import utils.ml_dataclass as dcl
    lambda_tv = res[dcl.MFbMLGD.key]['lambda']
    print("lambda_tv:", lambda_tv)

    iters_fine = ConfParam().iters_fine
    iters_coarse = 5
    if params_exp['noise_pow'] <= 0.05:  # on off system : computation time
        iters_coarse = 20
    iters_vec = _set_iter_vec(iters_coarse, iters_fine)
    #p_tv['iml_max_iter'] = 3
    p_tv['iml_max_iter'] = ConfParam().iml_max_iter

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

    return p_tv

def get_parameters_dpir(params_exp):
    return {}

def get_parameters_tv_coarse_pgd(params_exp):
    p_tv = get_parameters_tv(params_exp)
    p_tv['coarse_iterator'] = CPGDIteration
    return p_tv

# ============== multilevel specific modifiers ==============
def set_ml_param_Moreau(params, params_exp):
    params['coarse_iterator'] = CGDIteration
    iters_vec = params['params_multilevel'][0]['iters']
    gamma_vec = [1.1] * len(iters_vec)
    gamma_vec[-1] = 1.0
    params['params_multilevel'][0]['gamma_moreau'] = gamma_vec
    params['gamma_moreau'] = gamma_vec[-1]

    params['coherence_prior'] = CTV()
    params['coarse_prior'] = CTV()

    return params

def set_ml_param_student(params, params_exp):
    device = params_exp['device']
    denoiser = ConfParam().get_student(device)

    prior_class = params['prior'].__class__
    params['coherence_prior'] = prior_class(denoiser=denoiser)
    params['coarse_prior'] = prior_class(denoiser=denoiser)

    return params

def set_ml_param_noreg(params, params_exp):
    params['coherence_prior'] = params['prior']
    params['coarse_prior'] = Zero()
    return params


# ============== utility functions ==============
def set_new_nb_coarse(params):
    new_nb = 8
    l_iter = params['params_multilevel'][0]['iters'][0:-1]
    params['params_multilevel'][0]['iters'][0:-1] = [new_nb] * len(l_iter)
    print("iters_coarse:", new_nb)
def get_multilevel_init_params(params):
    iters_vec = params['params_multilevel'][0]['iters']
    param_init = params.copy()
    # does not iterate on finest level
    param_init['init_ml_x0'] = [80] * len(iters_vec)
    return param_init

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

    res = {}
    import utils.ml_dataclass as dcl
    alg = [
        dcl.MRedMLInit.key,
        dcl.MPnPML.key,
        dcl.MPnPProxML.key,
        dcl.MFbMLGD.key,
    ]

    print("def_noise:", noise_pow)
    if 'gridsearch' in params_exp.keys() and params_exp['gridsearch'] is True:
        for akey in alg:
            res[akey] = {'lambda': 0, 'g_param': 0}
    elif problem == 'inpainting' or problem == 'demosaicing':
        for akey in alg:
            res[akey] = inpainting_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'blur' or problem == 'motion_blur':
        for akey in alg:
            res[akey] = blur_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'mri':
        for akey in alg:
            res[akey] = blur_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'tomography':
        pass
    else:
        raise NotImplementedError("not implem")

    params_algo = {
        'cit': ConfParam().win,
        'scale_coherent_grad': True
    }

    return params_algo, res


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


#def get_parameters_red_approx(params_exp):
#    p_red = get_parameters_red(params_exp)
#    d = Student(layers=10, nc=32, cnext_ic=2, pretrained=state_file_v3).to(params_exp['device'])
#    p_red['coarse_prior'] = RED(denoiser=d)
#    p_red['coherence_prior'] = RED(denoiser=d)
#
#    return p_red
#
#def get_parameters_red_approx_noreg(params_exp):
#    p_red = get_parameters_red_approx(params_exp)
#    p_red['coherence_prior'] = p_red['coarse_prior']
#    p_red['coarse_prior'] = Zero()
#    return p_red
#
#def get_parameters_red_moreau(params_exp):
#    p_red = get_parameters_red(params_exp)
#    p_red['coarse_iterator'] = CGDIteration
#    iters_vec = p_red['params_multilevel'][0]['iters']
#    gamma_vec = [1.1] * len(iters_vec)
#    gamma_vec[-1] = 1.0
#    p_red['params_multilevel'][0]['gamma_moreau'] = gamma_vec
#    p_red['gamma_moreau'] = gamma_vec[-1]
#
#    p_red['coherence_prior'] = CTV()
#    p_red['coarse_prior'] = CTV()
#    return p_red

#def get_parameter_pnp_Moreau(params_exp):
#    p_pnp = get_parameters_pnp_prox(params_exp)
#    p_pnp['coarse_iterator'] = CGDIteration
#    iters_vec = p_pnp['params_multilevel'][0]['iters']
#    gamma_vec = [1.1] * len(iters_vec)
#    gamma_vec[-1] = 1.0
#    p_pnp['params_multilevel'][0]['gamma_moreau'] = gamma_vec
#    p_pnp['gamma_moreau'] = gamma_vec[-1]
#
#    p_pnp['coherence_prior'] = CTV()
#    p_pnp['coarse_prior'] = CTV()
#
#    return p_pnp

#def get_parameters_pnp_stud(params_exp):
#    p_pnp = get_parameters_pnp_prox(params_exp)
#
#    #state_file = os.path.join(checkpoint_path(), 'student_v1_5K_kersz1_KD_mse.pth.tar')
#    #d = Student0(layers=5, nc=64, cnext_ic=4, pretrained=state_file).to(params_exp['device'])
#    #state_file = os.path.join(checkpoint_path(), 'student_v3_cs_c32_ic2_10L.pth.tar')
#    #state_file = os.path.join(checkpoint_path(), 'student_v3_cs_c32_ic2_10L_525.pth.tar')
#    #state_file = os.path.join(checkpoint_path(), 'student_v4_cs_c32_ic2_10L_weight2_599.pth.tar')
#    d = Student(layers=10, nc=32, cnext_ic=2, pretrained=state_file_v3).to(params_exp['device'])
#
#    p_pnp['coherence_prior'] = PnP(denoiser=d)
#    p_pnp['coarse_prior'] = PnP(denoiser=d)
#
#    return p_pnp

#def get_parameters_pnp_prox_noreg(params_exp):
#    p_pnp = get_parameters_pnp_prox(params_exp)
#    device = params_exp['device']
#    p_pnp['coherence_prior'] = PnP(denoiser=GSDRUNet(pretrained="download", device=device))
#    p_pnp['coarse_prior'] = Zero()
#    return p_pnp
#
#def get_parameters_pnp_stud_noreg(params_exp):
#    p_pnp = get_parameters_pnp_stud(params_exp)
#    p_pnp['coherence_prior'] = p_pnp['coarse_prior']
#    p_pnp['coarse_prior'] = Zero()
#    return p_pnp

#def get_parameters_pnp_prox_nc(params_exp):
#    p_pnp = get_parameters_pnp_prox(params_exp)
#    p_pnp['scale_coherent_grad'] = False
#    return p_pnp
#
#def get_parameters_pnp_approx_nc(params_exp):
#    p_pnp = get_parameters_pnp_stud(params_exp)
#    p_pnp['scale_coherent_grad'] = False
#    return p_pnp