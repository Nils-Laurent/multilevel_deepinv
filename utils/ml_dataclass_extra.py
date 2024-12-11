from utils.parameters import *
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
    key = 'PnP_ML_stud_N1c'
    color = 'purple'
    linestyle = 'solid'
    label = key
    #param_fn = get_parameters_pnp_approx_nc
    param_fn = get_parameters_pnp_prox
    edit_fn = [set_ml_param_student]
    replace_params = {'scale_coherent_grad': False}
@dataclass
class MPnPMLNc:
    key = 'PnP_ML_N1c'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_pnp_prox
    replace_params = {'scale_coherent_grad': False}
@dataclass
class MPnPMLNoR:
    key = 'PnP_ML_NoReg'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_pnp_prox
    edit_fn = [set_ml_param_noreg]
    #param_fn = get_parameters_pnp_prox_noreg