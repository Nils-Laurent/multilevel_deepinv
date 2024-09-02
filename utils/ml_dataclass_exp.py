from tests.parameters import *
from dataclasses import dataclass


@dataclass
class MRedInit:
    key = 'RED_INIT'
    color = 'black'
    linestyle = 'dashed'
    label = key
    param_fn = get_parameters_red
    use_init = True
@dataclass
class MPnPMLApproxNc:
    key = 'PnP_ML_approx_N1c'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_pnp_approx_nc
@dataclass
class MPnPMLNc:
    key = 'PnP_ML_N1c'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_pnp_prox_nc
@dataclass
class MPnPMLNoR:
    key = 'PnP_ML_NoReg'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_pnp_prox_noreg