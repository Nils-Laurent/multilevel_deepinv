import tests.parameters as param
from dataclasses import dataclass


# ============= PnP =============
@dataclass
class MPnP:
    key = 'PnP'
    color = 'purple'
    linestyle = 'dashed'
    single_level = True
    label = key
    param_fn = param.get_parameters_pnp_drunet
@dataclass
class MPnPInit:
    key = 'PnP_INIT'
    color = 'purple'
    linestyle = 'dashed'
    single_level = True
    label = key
    use_init = True
    param_fn = param.get_parameters_pnp_drunet
@dataclass
class MPnPML:
    key = 'PnP_ML'
    color = 'gray'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_drunet
@dataclass
class MPnPMLStud:
    key = 'PnP_ML_stud'
    color = 'gray'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_drunet
    edit_fn = [param.set_ml_param_student]
@dataclass
class MPnPMLStudNoR:
    key = 'PnP_ML_stud_NoReg'
    color = 'gray'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_drunet
    edit_fn = [param.set_ml_param_student, param.set_ml_param_noreg]
@dataclass
class MPnPMoreau:
    key = 'PnP_ML_Moreau'
    color = 'gray'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_drunet
    edit_fn = [param.set_ml_param_Moreau]
@dataclass
class MPnPMLInit:
    key = 'PnP_ML_INIT'
    color = 'gray'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_drunet
    use_init = True
@dataclass
class MPnPMLStudInit:
    key = 'PnP_ML_stud_INIT'
    color = 'gray'
    linestyle = 'solid'
    label = key
    use_init = True
    param_fn = param.get_parameters_pnp_drunet
    edit_fn = [param.set_ml_param_student]
@dataclass
class MPnPMLStudNoRInit:
    key = 'PnP_ML_stud_NoReg_INIT'
    color = 'gray'
    linestyle = 'solid'
    label = key
    use_init = True
    param_fn = param.get_parameters_pnp_drunet
    edit_fn = [param.set_ml_param_student, param.set_ml_param_noreg]
@dataclass
class MPnPMoreauInit:
    key = 'PnP_ML_Moreau_INIT'
    color = 'gray'
    linestyle = 'solid'
    label = key
    use_init = True
    param_fn = param.get_parameters_pnp_drunet
    edit_fn = [param.set_ml_param_Moreau]

# ============= PnP prox =============
@dataclass
class MPnPProx:
    key = 'PnP_prox'
    color = 'purple'
    linestyle = 'dashed'
    single_level = True
    label = key
    param_fn = param.get_parameters_pnp_prox
@dataclass
class MPnPProxInit:
    key = 'PnP_prox_INIT'
    color = 'purple'
    linestyle = 'dashed'
    single_level = True
    label = key
    param_fn = param.get_parameters_pnp_prox
    use_init = True
@dataclass
class MPnPProxML:
    key = 'PnP_prox_ML'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_prox
@dataclass
class MPnPProxMoreau:
    key = 'PnP_prox_ML_Moreau'
    color = 'purple'
    linestyle = 'dashed'
    label = key
    param_fn = param.get_parameters_pnp_prox
    edit_fn = [param.set_ml_param_Moreau]
@dataclass
class MPnPProxMLStud:
    key = 'PnP_prox_ML_stud'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_prox
    edit_fn = [param.set_ml_param_student]
@dataclass
class MPnPProxMLInit:
    key = 'PnP_prox_ML_INIT'
    color = 'purple'
    linestyle = 'solid'
    label = key
    use_init = True
    param_fn = param.get_parameters_pnp_prox
@dataclass
class MPnPProxMoreauInit:
    key = 'PnP_prox_ML_Moreau_INIT'
    color = 'purple'
    linestyle = 'dashed'
    label = key
    use_init = True
    param_fn = param.get_parameters_pnp_prox
    edit_fn = [param.set_ml_param_Moreau]
@dataclass
class MPnPProxMLStudInit:
    key = 'PnP_prox_ML_stud_INIT'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_prox
    edit_fn = [param.set_ml_param_student]
    use_init = True
@dataclass
class MPnPProxMLStudNoR:
    key = 'PnP_prox_ML_stud_NoReg'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_prox
    edit_fn = [param.set_ml_param_student, param.set_ml_param_noreg]
@dataclass
class MPnPProxMLStudNoRInit:
    key = 'PnP_prox_ML_stud_NoReg_INIT'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_pnp_prox
    edit_fn = [param.set_ml_param_student, param.set_ml_param_noreg]
    use_init = True

# ============= RED =============
@dataclass
class MRed:
    key = 'RED'
    color = 'red'
    linestyle = 'dashed'
    single_level = True
    label = key
    param_fn = param.get_parameters_red
@dataclass
class MRedInit:
    key = 'RED_INIT'
    color = 'red'
    linestyle = 'dashed'
    single_level = True
    label = key
    use_init = True
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
@dataclass
class MRedMLStud:
    key = 'RED_ML_stud'
    color = 'black'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_red
    edit_fn = [param.set_ml_param_student]
@dataclass
class MRedMLStudNoR:
    key = 'RED_ML_stud_NoReg'
    color = 'black'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_red
    edit_fn = [param.set_ml_param_student, param.set_ml_param_noreg]
@dataclass
class MRedMLStudNoRInit:
    key = 'RED_ML_stud_NoReg_INIT'
    color = 'black'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_red
    use_init = True
    edit_fn = [param.set_ml_param_student, param.set_ml_param_noreg]
@dataclass
class MRedMLStudInit:
    key = 'RED_ML_stud_INIT'
    color = 'black'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_red
    edit_fn = [param.set_ml_param_student]
    use_init = True
@dataclass
class MRedMLMoreau:
    key = 'RED_ML_Moreau'
    color = 'black'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_red
    edit_fn = [param.set_ml_param_Moreau]
@dataclass
class MRedMLMoreauInit:
    key = 'RED_ML_Moreau_INIT'
    color = 'black'
    linestyle = 'solid'
    label = key
    use_init = True
    param_fn = param.get_parameters_red
    edit_fn = [param.set_ml_param_Moreau]

# ============= TV =============
@dataclass
class MFb:
    key = 'FB_TV'
    color = 'blue'
    linestyle = 'dashed'
    single_level = True
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
class MDPIRLong:
    key = 'DPIR_Long'
    color = 'green'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_dpir

@dataclass
class MDPIR:
    key = 'DPIR'
    color = 'green'
    linestyle = 'solid'
    label = key
    param_fn = param.get_parameters_dpir