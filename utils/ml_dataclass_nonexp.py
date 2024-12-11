import utils.parameters as param
from dataclasses import dataclass

# ============= PnP NE =============
@dataclass
class MPnPNE:
    key = 'PnP_NE'
    color = 'purple'
    linestyle = 'dashed'
    single_level = True
    label = key
    param_fn = param.get_parameters_pnp_non_exp
@dataclass
class MPnPNEInit:
    key = 'PnP_NE_INIT'
    color = 'purple'
    linestyle = 'dashed'
    single_level = True
    label = key
    param_fn = param.get_parameters_pnp_non_exp
    use_init = True
@dataclass
class MPnPNEML:
    key = 'PnP_NE_ML'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_non_exp
@dataclass
class MPnPNEMoreau:
    key = 'PnP_NE_ML_Moreau'
    color = 'purple'
    linestyle = 'dashed'
    label = key
    param_fn = param.get_parameters_pnp_non_exp
    edit_fn = [param.set_ml_param_Moreau]
@dataclass
class MPnPNEMLStud:
    key = 'PnP_NE_ML_stud'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_non_exp
    edit_fn = [param.set_ml_param_student]
@dataclass
class MPnPNEMLInit:
    key = 'PnP_NE_ML_INIT'
    color = 'purple'
    linestyle = 'solid'
    label = key
    use_init = True
    param_fn = param.get_parameters_pnp_non_exp
@dataclass
class MPnPNEMoreauInit:
    key = 'PnP_NE_ML_Moreau_INIT'
    color = 'purple'
    linestyle = 'dashed'
    label = key
    use_init = True
    param_fn = param.get_parameters_pnp_non_exp
    edit_fn = [param.set_ml_param_Moreau]
@dataclass
class MPnPNEMLStudInit:
    key = 'PnP_NE_ML_stud_INIT'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_non_exp
    edit_fn = [param.set_ml_param_student]
    use_init = True
