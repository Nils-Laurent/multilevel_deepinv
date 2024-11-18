import os

import deepinv
from deepinv.optim import L2
from deepinv.optim.prior import ScorePrior, RED, PnP, TVPrior, Zero
from deepinv.models import DRUNet, GSDRUNet, DnCNN

from multilevel.approx_nn import Student
from multilevel.coarse_gradient_descent import CGDIteration
from multilevel.coarse_pgd import CPGDIteration
from multilevel.info_transfer import BlackmannHarris, CFir, SincFilter
from multilevel.prior import TVPrior as CustTV
from multilevel_utils.custom_poisson_noise import CPoissonLikelihood
from utils.get_hyper_param import inpainting_hyper_param, blur_hyper_param, mri_hyper_param, poisson_hyper_param
from utils.paths import checkpoint_path

from multilevel_utils.complex_denoiser import to_complex_denoiser


state_file_v3 = os.path.join(checkpoint_path(), 'student_v3_cs_c32_ic2_10L_525.pth.tar')
state_file_v4 = os.path.join(checkpoint_path(), 'student_v4_cs_c32_ic2_10L_weight2_599.pth.tar')

#state_file_1channel = os.path.join(checkpoint_path(), 'student_1channel_ckp_599.pth.tar')
state_file_1channel = os.path.join(checkpoint_path(), '24-09-20-12:41:35/ckp_599.pth.tar')


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ConfParam(metaclass=Singleton):
    win = None
    levels = None
    iters_fine = None
    iml_max_iter = None
    coarse_iters_ini = None
    use_complex_denoiser = None
    data_fidelity = None
    data_fidelity_lipschitz = None
    denoiser_in_channels = None
    s1coherent_algorithm = None
    s1coherent_init = None
    iter_coarse_pnp_map = None
    iter_coarse_pnp_pgd = None
    iter_coarse_tv = None
    iter_coarse_red = None
    stepsize_multiplier_pnp = None

    def reset(self):
        self.win = SincFilter()
        self.levels = 4
        self.iters_fine = 200
        self.iml_max_iter = 2
        self.coarse_iters_ini = 5
        self.use_complex_denoiser = False
        self.data_fidelity = L2
        self.data_fidelity_lipschitz = 1.0  # data-fidelity Lipschitz cst
        self.denoiser_in_channels = 3
        self.s1coherent_algorithm = True
        self.s1coherent_init = False
        self.iter_coarse_pnp_map = 3
        self.iter_coarse_pnp_pgd = 3
        self.iter_coarse_tv = 3
        self.iter_coarse_red = 3
        self.stepsize_multiplier_pnp = 1.0

    def get_drunet(self, device):
        net = DRUNet(in_channels=self.denoiser_in_channels, out_channels=self.denoiser_in_channels, pretrained="download", device=device)
        denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        if self.use_complex_denoiser is True:
            denoiser = to_complex_denoiser(denoiser, mode="separated")
        denoiser.eval()
        return denoiser

    def get_dncnn_nonexp(self, device):
        net = DnCNN(
            in_channels=self.denoiser_in_channels, out_channels=self.denoiser_in_channels, pretrained="download_lipschitz", device=device
        )
        denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        denoiser.eval()
        return denoiser

    def get_student(self, device):
        d = Student(layers=10, nc=32, cnext_ic=2, pretrained=state_file_v3).to(device)
        if ConfParam().use_complex_denoiser is True:
            d = to_complex_denoiser(d, mode="separated")
        d.eval()
        return d

    def get_student1c(self, device):
        d = Student(in_channels=self.denoiser_in_channels,
                    layers=10, nc=32, cnext_ic=2, pretrained=state_file_1channel).to(device)
        if ConfParam().use_complex_denoiser is True:
            d = to_complex_denoiser(d, mode="separated")
        d.eval()
        return d

    def get_gsdrunet(self, device):
        net = GSDRUNet(pretrained="download", device=device)
        denoiser = net
        #denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        if ConfParam().use_complex_denoiser is True:
            denoiser = to_complex_denoiser(denoiser, mode="separated")
        denoiser.eval()
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
    import utils.ml_dataclass as dcl
    key_vec = [dcl.MPnPMLInit.key, dcl.MPnPMoreauInit.key]
    params_algo, res = get_param_algo_(params_exp, key_vec)
    p_pnp = params_algo.copy()

    p_pnp['g_param'] = res[dcl.MPnPMLInit.key]['g_param']

    # in case the smooth approx. of TV is used in coarse scales
    lambda_pnp = res[dcl.MPnPMoreauInit.key]['lambda']

    device = params_exp['device']
    denoiser = ConfParam().get_drunet(device)
    p_pnp['prior'] = PnP(denoiser=denoiser)
    p_pnp['prior'].eval()

    iters_fine = ConfParam().iters_fine
    iters_coarse = ConfParam().iter_coarse_pnp_map
    iters_vec = _set_iter_vec(iters_coarse, iters_fine)
    p_pnp['iml_max_iter'] = ConfParam().iml_max_iter

    p_pnp = standard_multilevel_param(p_pnp, it_vec=iters_vec, lambda_fine=lambda_pnp)
    p_pnp['coarse_iterator'] = CPGDIteration
    p_pnp['lip_g'] = prior_lipschitz(PnP, p_pnp, DRUNet)

    lambda_vec = p_pnp['params_multilevel'][0]['lambda']
    lf = ConfParam().data_fidelity_lipschitz

    step_size_coeff = ConfParam().stepsize_multiplier_pnp * 0.9
    stepsize_vec = [step_size_coeff/lf] * ConfParam().levels # PGD : only depends on lipschitz of data-fidelity

    p_pnp = _finalize_params(p_pnp, lambda_vec, stepsize_vec)
    return p_pnp

def get_parameters_pnp_non_exp(params_exp):
    import utils.ml_dataclass as dcl
    key_vec = [dcl.MPnPMLInit.key]
    params_algo, res = get_param_algo_(params_exp, key_vec)
    p_pnp = params_algo.copy()

    p_pnp['g_param'] = res[dcl.MPnPMLInit.key]['g_param']
    lambda_pnp = res[dcl.MPnPMLInit.key]['lambda']
    p_pnp['g_param'] = 0.2
    #p_pnp['g_param'] = 0.8
    #lambda_pnp = 0.2
    print("lambda NE :", lambda_pnp)
    print("g_param NE :", p_pnp['g_param'])

    device = params_exp['device']
    denoiser = ConfParam().get_dncnn_nonexp(device)
    p_pnp['prior'] = PnP(denoiser=denoiser)
    p_pnp['prior'].eval()

    iters_fine = ConfParam().iters_fine
    iters_coarse = ConfParam().iter_coarse_pnp_map
    iters_vec = _set_iter_vec(iters_coarse, iters_fine)
    p_pnp['iml_max_iter'] = ConfParam().iml_max_iter

    p_pnp = standard_multilevel_param(p_pnp, it_vec=iters_vec, lambda_fine=lambda_pnp)
    p_pnp['coarse_iterator'] = CPGDIteration
    p_pnp['lip_g'] = 1.0

    lambda_vec = p_pnp['params_multilevel'][0]['lambda']
    lf = ConfParam().data_fidelity_lipschitz

    stepsize_vec = [0.9/lf for l in lambda_vec] # PGD : only depends on lipschitz of data-fidelity
    stepsize_vec[-1] = 0.9/lf  # PGD : only depends on lipschitz of data-fidelity

    p_pnp = _finalize_params(p_pnp, lambda_vec, stepsize_vec)
    return p_pnp

def get_parameters_pnp_prox(params_exp):
    import utils.ml_dataclass as dcl
    key_vec = [dcl.MPnPProxMLInit.key]
    params_algo, res = get_param_algo_(params_exp, key_vec)
    p_pnp = params_algo.copy()

    p_pnp['g_param'] = res[dcl.MPnPProxMLInit.key]['g_param']
    lambda_pnp = 2.0 * ConfParam().data_fidelity_lipschitz /3.0
    print("lambda_pnp_prox:", lambda_pnp)

    device = params_exp['device']
    denoiser = ConfParam().get_gsdrunet(device)
    p_pnp['prior'] = PnP(denoiser=denoiser)

    iters_fine = ConfParam().iters_fine
    iters_coarse = ConfParam().iter_coarse_pnp_pgd
    print("iters_coarse:", iters_coarse)
    iters_vec = _set_iter_vec(iters_coarse, iters_fine)
    p_pnp['iml_max_iter'] = ConfParam().iml_max_iter

    p_pnp = standard_multilevel_param(p_pnp, it_vec=iters_vec, lambda_fine=lambda_pnp)
    p_pnp['coarse_iterator'] = CPGDIteration
    p_pnp['lip_g'] = prior_lipschitz(PnP, p_pnp, GSDRUNet)
    p_pnp['backtracking'] = False

    lambda_vec = [lambda_pnp]  * ConfParam().levels

    stepsize_vec = [1.0/lambda_pnp] * (ConfParam().levels - 1)

    # CANNOT CHOOSE STEPSIZE : see S. Hurault Thesis, Theorem 19.
    # lambda_pnp > 2 * data_fidelity_lipschitz / 3
    stepsize_vec.append(1.0/lambda_pnp)

    p_pnp = _finalize_params(p_pnp, lambda_vec, stepsize_vec)
    return p_pnp


def get_parameters_red(params_exp):
    import utils.ml_dataclass as dcl
    key_vec = [dcl.MRedMLInit.key]
    params_algo, res = get_param_algo_(params_exp, key_vec)
    p_red = params_algo.copy()

    p_red['g_param'] = res[dcl.MRedMLInit.key]['g_param']
    lambda_red = res[dcl.MRedMLInit.key]['lambda']
    print("lambda_red:", lambda_red)

    device = params_exp['device']
    denoiser = ConfParam().get_drunet(device)
    p_red['prior'] = RED(denoiser=denoiser)
    p_red['prior'].eval()

    iters_fine = ConfParam().iters_fine
    iters_coarse = ConfParam().iter_coarse_red
    iters_vec = _set_iter_vec(iters_coarse, iters_fine)
    p_red['iml_max_iter'] = ConfParam().iml_max_iter

    p_red = standard_multilevel_param(p_red, it_vec=iters_vec, lambda_fine=lambda_red)
    p_red['lip_g'] = prior_lipschitz(RED, p_red, DRUNet)

    #p_red['params_multilevel'][0]['lambda'] = [lambda_red] * ConfParam().levels
    lambda_vec = p_red['params_multilevel'][0]['lambda']
    step_coeff = 0.9  # non-convex setting
    lf = ConfParam().data_fidelity_lipschitz
    stepsize_vec = [step_coeff / (lf + l * p_red['lip_g']) for l in lambda_vec] # gradient descent

    p_red = _finalize_params(p_red, lambda_vec=lambda_vec, stepsize_vec=stepsize_vec)
    return p_red

def get_parameters_tv(params_exp):
    import utils.ml_dataclass as dcl
    key_vec = [dcl.MFbMLGD.key]
    # We assume regularization gradient is 1-Lipschitz
    params_algo, res = get_param_algo_(params_exp, key_vec)
    params_algo['scale_coherent_grad'] = True  # for FB TV we always use 1order coherence
    p_tv = params_algo.copy()

    lambda_tv = res[dcl.MFbMLGD.key]['lambda']
    print("lambda_tv:", lambda_tv)

    iters_fine = ConfParam().iters_fine
    iters_coarse = ConfParam().iter_coarse_tv
    iters_vec = _set_iter_vec(iters_coarse, iters_fine)
    p_tv['iml_max_iter'] = ConfParam().iml_max_iter

    p_tv = standard_multilevel_param(p_tv, it_vec=iters_vec, lambda_fine=lambda_tv)
    p_tv['scale_coherent_grad'] = True  # for FB TV we always use 1order coherence
    p_tv['lip_g'] = prior_lipschitz(TVPrior, p_tv)
    p_tv['prox_crit'] = 1e-6
    p_tv['prox_max_it'] = 1000
    gamma_vec = [1.1] * len(iters_vec)
    gamma_vec[-1] = 1.0
    lambda_vec = p_tv['params_multilevel'][0]['lambda']
    lf = ConfParam().data_fidelity_lipschitz
    step_coeff = 1.9  # convex setting
    stepsize_vec = [step_coeff / (lf + 1.0/gamma) for gamma in gamma_vec]
    stepsize_vec[-1] = step_coeff / lf  # only depends on lipschitz of data-fidelity
    p_tv = _finalize_params(p_tv, lambda_vec, stepsize_vec, gamma_vec)
    p_tv['scale_coherent_grad'] = True  # for FB TV we always use 1order coherence

    return p_tv

def get_parameters_dpir(params_exp):
    return {}

def get_parameters_tv_coarse_pgd(params_exp):
    p_tv = get_parameters_tv(params_exp)
    p_tv['coarse_iterator'] = CPGDIteration
    return p_tv

# ============== multilevel specific modifiers ==============
def set_ml_param_Moreau(params, params_exp):
    if isinstance(params['prior'], PnP):
        import utils.ml_dataclass as dcl
        key_vec = [dcl.MPnPMoreauInit.key]
        params_algo, res = get_param_algo_(params_exp, key_vec)
        params['g_param'] = res[dcl.MPnPMoreauInit.key]['g_param']

    params['coarse_iterator'] = CGDIteration
    iters_vec = params['params_multilevel'][0]['iters']
    gamma_vec = [1.1] * len(iters_vec)
    gamma_vec[-1] = 1.0
    params['params_multilevel'][0]['gamma_moreau'] = gamma_vec
    params['gamma_moreau'] = gamma_vec[-1]

    # todo : A VALIDER
    #params['coherence_prior'] = CustTV()
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
    # todo : A VALIDER
    #params['coherence_prior'] = prior_class(denoiser=denoiser)
    params['coarse_prior'] = prior_class(denoiser=denoiser)

    return params

def set_ml_param_noreg(params, params_exp):
    assert False  # normally, it is not used anymore
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
    params['params_multilevel'][0]['iters_init'] = [ConfParam().coarse_iters_ini] * len(iters_vec)
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

    #params['multilevel_step'][100 + -1] = True
    #params['multilevel_step'][100 + -2] = True

    return params


def single_level_params(params_ml):
    params = params_ml.copy()
    params['n_levels'] = 1
    params['level'] = 1
    params['iters'] = params_ml['params_multilevel'][0]['iters'][-1]
    params['lambda'] = params_ml['params_multilevel'][0]['lambda'][-1]
    params['stepsize'] = params_ml['params_multilevel'][0]['stepsize'][-1]

    return params


def get_param_algo_(params_exp, key_vec):
    noise_pow = params_exp["noise_pow"]
    problem = params_exp['problem']

    res = {}

    print("def_noise:", noise_pow)
    if 'gridsearch' in params_exp.keys() and params_exp['gridsearch'] is True:
        for akey in key_vec:
            res[akey] = {'lambda': 0, 'g_param': 0}
    elif problem == 'inpainting' or problem == 'demosaicing':
        for akey in key_vec:
            res[akey] = inpainting_hyper_param(noise_pow=noise_pow, gs_key=akey)
    elif problem == 'blur' or problem == 'motion_blur':
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

    params_algo = {
        'cit': ConfParam().win,
        'scale_coherent_grad': ConfParam().s1coherent_algorithm,
        'scale_coherent_grad_init': ConfParam().s1coherent_init,
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

