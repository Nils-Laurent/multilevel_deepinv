import utils.parameters as param
from dataclasses import dataclass


# ============= PnP =============
@dataclass
class MPnPDnCNN:
    key = 'PnP_DnCNN'
    color = 'purple'
    linestyle = 'dashed'
    single_level = True
    label = key
    param_fn = param.get_parameters_pnp_dncnn
@dataclass
class MPnPMLDnCNNInit:
    key = 'PnP_ML_DnCNN_init'
    color = 'gray'
    linestyle = 'solid'
    label = key
    use_init = True
    param_fn = param.get_parameters_pnp_dncnn
@dataclass
class MPnPMLDnCNNMoreauInit:
    key = 'PnP_ML_DnCNN_Moreau_init'
    color = 'gray'
    linestyle = 'solid'
    label = key
    use_init = True
    param_fn = param.get_parameters_pnp_dncnn
    edit_fn = [param.set_ml_param_Moreau]
@dataclass
class MPnPSCUNet:
    key = 'PnP_SCUNet'
    color = 'purple'
    linestyle = 'dashed'
    single_level = True
    label = key
    param_fn = param.get_parameters_pnp_scunet
@dataclass
class MPnPMLSCUNetInit:
    key = 'PnP_ML_SCUNet_init'
    color = 'gray'
    linestyle = 'solid'
    label = key
    use_init = True
    param_fn = param.get_parameters_pnp_scunet
@dataclass
class MPnPMLSCUNetMoreauInit:
    key = 'PnP_ML_SCUNet_Moreau_init'
    color = 'gray'
    linestyle = 'solid'
    label = key
    use_init = True
    param_fn = param.get_parameters_pnp_scunet
    edit_fn = [param.set_ml_param_Moreau]
