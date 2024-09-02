import tests.parameters as param
from dataclasses import dataclass

# ============= PnP =============
@dataclass
class MPnP:
    key = 'PnP'
    color = 'purple'
    linestyle = 'dashed'
    label = key
    param_fn = param.get_parameters_pnp_prox
@dataclass
class MPnPML:
    key = 'PnP_ML'
    color = 'gray'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp
@dataclass
class MPnPMLInit:
    key = 'PnP_ML_INIT'
    color = 'gray'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp
    use_init = True
@dataclass
class MPnPMLProx:
    key = 'PnP_ML_prox'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_prox
@dataclass
class MPnPMoreau:
    key = 'PnP_Moreau'
    color = 'purple'
    linestyle = 'dashed'
    label = key
    param_fn = param.get_parameter_pnp_Moreau
@dataclass
class MPnPMLApprox:
    key = 'PnP_ML_approx'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_approx
@dataclass
class MPnPMLApproxInit:
    key = 'PnP_ML_approx_INIT'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_approx
    use_init = True
@dataclass
class MPnPMLApproxNoR:
    key = 'PnP_ML_approx_NoReg'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_approx_noreg

# ============= RED =============
@dataclass
class MRed:
    key = 'RED'
    color = 'red'
    linestyle = 'dashed'
    label = key
    param_fn = param.get_parameters_red
@dataclass
class MRedML:
    key = 'RED_ML'
    color = 'red'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_red
@dataclass
class MRedMLInit:
    key = 'RED_ML_INIT'
    color = 'black'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_red
    use_init = True

# ============= TV =============
@dataclass
class MFb:
    key = 'FB_TV'
    color = 'blue'
    linestyle = 'dashed'
    label = key
    param_fn = param.get_parameters_tv
@dataclass
class MFbMLGD:
    key = 'FB_TV_ML'
    color = 'blue'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_tv
@dataclass
class MFbMLProx:
    key = 'FB_TV_ML_prox'
    color = 'blue'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_tv_coarse_pgd

# ============= Others =============
@dataclass
class MDPIR:
    key = 'DPIR'
    color = 'green'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_dpir