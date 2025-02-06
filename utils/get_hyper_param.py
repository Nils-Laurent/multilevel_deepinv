from utils.parameters_global import ConfParam


def select_param(noise_pow, noise_vec, p_red, p_pnp, p_tv):
    for n in range(len(noise_vec)):
        if noise_vec[n] == noise_pow:
            p_red = p_red[n]
            p_pnp = p_pnp[n]
            p_tv = p_tv[n]
            break

    if type(p_red) is not dict:
        raise NotImplementedError()

    return p_red, p_pnp, p_tv

def affine_interpolation(bounds_data, noise_pow):
    # bounds_data[0] follows [NOISE, KEY, PARAM]
    assert len(bounds_data) == 2

    n0 = bounds_data[0][0]
    n1 = bounds_data[1][0]
    assert n0 <= noise_pow <= n1

    x = (noise_pow - n0) / (n1 - n0)

    param_out = {}
    d0 = bounds_data[0][2]
    d1 = bounds_data[1][2]
    for k_ in d0.keys():
        param_out[k_] = (1 - x) * d0[k_] + x * d1[k_]

    return param_out

def gs_pick_bounds(gs_vec, gs_key, noise_pow):
    # gs_vec[0] follows [NOISE, KEY, PARAM]
    # gs_vec is sorted w.r.t. dim zero (NOISE)
    data = []
    for el in gs_vec:
        if el[1] == gs_key:
            data.append(el)

    assert len(data) > 0, "No data found: key error ?"
    assert noise_pow >= data[0][0], "noise_pow smaller than gridsearch minimum"
    assert noise_pow <= data[-1][0], "noise_pow bigger than gridsearch maximum"

    if len(data) == 1:
        return data[0][2]  # return param dict

    id = 1
    while noise_pow > data[id][0]:
        id = id + 1

    return [data[id - 1], data[id]]

def blur_hyper_param(noise_pow, gs_key):
    gs_vec = []

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)

def poisson_hyper_param(noise_pow, gs_key):
    gs_vec = [
        [0.1, 'PnP_ML_INIT', {'g_param':0.0061, 'stepsz_coeff':1.89, }],  # PSNR = 18.66
        [0.1, 'PnP_ML_Moreau_INIT', {'lambda':0.306, 'g_param':0.01, 'stepsz_coeff':1.0, }],  # PSNR = 18.06
        [0.1, 'FB_TV_ML', {'lambda':0.25, }],  # PSNR = 19.81
        [0.1, 'PnP_ML_DnCNN_Moreau_init', {'lambda':0.5, 'stepsz_coeff':1.0, }],  # PSNR = 16.81
        [0.1, 'PnP_ML_SCUNet_Moreau_init', {'lambda':1.17, 'stepsz_coeff':0.333, }],  # PSNR = 7.83
        [0.1, 'PnP_prox_ML_INIT', {'g_param':0.0031, 'stepsz_coeff':1.67, }],  # PSNR = 19.31
        [0.1, 'PnP_prox_ML_Moreau_INIT', {'lambda':0.778, 'g_param':0.01, 'stepsz_coeff':0.75, }],  # PSNR = 18.72
    ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)


def inpainting_hyper_param(noise_pow, gs_key):
    gs_vec = [
        [0.1, 'FB_TV_ML', {'lambda':0.125, }],  # PSNR = 23.45
        [0.1, 'PnP', {'g_param':0.21, 'stepsz_coeff':2.0, }],  # PSNR = 27.73
        [0.1, 'PnP_INIT', {'g_param':0.11, 'stepsz_coeff':1.5, }],  # PSNR = 29.19
        [0.1, 'PnP_ML', {'g_param':0.16, 'stepsz_coeff':1.5, }],  # PSNR = 27.99
        [0.1, 'PnP_ML_INIT', {'g_param':0.11, 'stepsz_coeff':1.5, }],  # PSNR = 29.24
        [0.1, 'PnP_ML_Moreau', {'lambda':0.563, 'g_param':0.188, 'stepsz_coeff':1.13, }],  # PSNR = 25.70
        [0.1, 'PnP_ML_Moreau_INIT', {'lambda':0.188, 'g_param':0.0812, 'stepsz_coeff':1.13, }],  # PSNR = 28.99
        [0.1, 'PnP_prox', {'g_param':0.14, 'stepsz_coeff':1.75, }],  # PSNR = 28.90
        [0.1, 'PnP_prox_ML_INIT', {'g_param':0.12, 'stepsz_coeff':1.5, }],  # PSNR = 29.06
        [0.1, 'PnP_DnCNN', {'stepsz_coeff':2.58, }],  # PSNR = nan
        [0.1, 'PnP_ML_DnCNN_init', {'stepsz_coeff':2.5, }],  # PSNR = nan
        [0.1, 'PnP_SCUNet', {'stepsz_coeff':2.0, }],  # PSNR = 22.24
        [0.1, 'PnP_ML_SCUNet_init', {'stepsz_coeff':1.5, }],  # PSNR = 25.50
        #[0.1, 'FB_TV_ML', {'lambda':0.125, }],  # PSNR = 23.45
        #[0.1, 'PnP', {'g_param':0.2, 'stepsz_coeff':1.83, }],  # PSNR = 27.81
        #[0.1, 'PnP_ML_INIT', {'g_param':0.0701, 'stepsz_coeff':0.833, }],  # PSNR = 29.17
        #[0.1, 'PnP_ML_Moreau_INIT', {'lambda':2.44, 'g_param':0.0812, 'stepsz_coeff':1.13, }],  # PSNR = 29.12
        #[0.1, 'PnP_SCUNet', {'stepsz_coeff':2.0, }],  # PSNR = 22.84
        #[0.1, 'PnP_ML_SCUNet_init', {'stepsz_coeff':1.67, }],  # PSNR = 25.50
        #[0.1, 'PnP_prox', {'g_param':0.13, 'stepsz_coeff':1.62, }],  # PSNR = 29.06
        #[0.1, 'PnP_prox_ML_INIT', {'g_param':0.12, 'stepsz_coeff':1.5, }],  # PSNR = 29.12
    ]
    if ConfParam().use_equivariance is True:
        gs_vec = [
            [0.1, 'FB_TV_ML', {'lambda':0.125, }],  # PSNR = 23.43
            [0.1, 'PnP', {'g_param':0.21, 'stepsz_coeff':2.0, }],  # PSNR = 27.78
            [0.1, 'PnP_INIT', {'g_param':0.12, 'stepsz_coeff':1.67, }],  # PSNR = 29.29
            [0.1, 'PnP_ML', {'g_param':0.16, 'stepsz_coeff':1.5, }],  # PSNR = 28.05
            [0.1, 'PnP_ML_INIT', {'g_param':0.0801, 'stepsz_coeff':1.0, }],  # PSNR = 29.20
            [0.1, 'PnP_ML_Moreau', {'lambda':1.88, 'g_param':0.188, 'stepsz_coeff':2.44, }],  # PSNR = 25.75
            [0.1, 'PnP_ML_Moreau_INIT', {'lambda':2.44, 'g_param':0.0812, 'stepsz_coeff':0.75, }],  # PSNR = 29.10
            [0.1, 'PnP_prox', {'g_param':0.12, 'stepsz_coeff':1.62, }],  # PSNR = 29.06
            [0.1, 'PnP_prox_ML_INIT', {'g_param':0.11, 'stepsz_coeff':1.38, }],  # PSNR = 29.11
            [0.1, 'PnP_SCUNet', {'stepsz_coeff':2.0, }],  # PSNR = 22.44
            [0.1, 'PnP_ML_SCUNet_init', {'stepsz_coeff':1.67, }],  # PSNR = 25.53
        ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)

def demosaicing_hyper_param(noise_pow, gs_key):
    gs_vec = [
        [0.1, 'FB_TV_ML', {'lambda':0.0833, }],  # PSNR = 22.20
        [0.1, 'PnP', {'g_param':0.14, 'stepsz_coeff':1.83, }],  # PSNR = 28.38
        [0.1, 'PnP_INIT', {'g_param':0.14, 'stepsz_coeff':2.5, }],  # PSNR = 29.42
        [0.1, 'PnP_ML', {'g_param':0.14, 'stepsz_coeff':1.67, }],  # PSNR = 28.00
        [0.1, 'PnP_ML_INIT', {'g_param':0.0801, 'stepsz_coeff':1.17, }],  # PSNR = 29.13
        [0.1, 'PnP_ML_Moreau', {'lambda':1e-05, 'g_param':0.129, 'stepsz_coeff':0.375, }],  # PSNR = 26.27
        [0.1, 'PnP_ML_Moreau_INIT', {'lambda':2.25, 'g_param':0.0813, 'stepsz_coeff':1.13, }],  # PSNR = 28.42
        [0.1, 'PnP_prox', {'g_param':0.28, 'stepsz_coeff':1.88, }],  # PSNR = 26.31
        [0.1, 'PnP_prox_ML_INIT', {'g_param':0.1, 'stepsz_coeff':1.5, }],  # PSNR = 29.12
        [0.1, 'PnP_DnCNN', {'stepsz_coeff':2.58, }],  # PSNR = nan
        [0.1, 'PnP_ML_DnCNN_init', {'stepsz_coeff':2.58, }],  # PSNR = nan
        [0.1, 'PnP_SCUNet', {'stepsz_coeff':2.33, }],  # PSNR = 25.09
        [0.1, 'PnP_ML_SCUNet_init', {'stepsz_coeff':2.0, }],  # PSNR = 25.02
        #[0.1, 'FB_TV_ML', {'lambda':0.0833, }],  # PSNR = 22.19
        #[0.1, 'PnP', {'g_param':0.15, 'stepsz_coeff':2.0, }],  # PSNR = 28.49
        #[0.1, 'PnP_ML_INIT', {'g_param':0.0801, 'stepsz_coeff':1.17, }],  # PSNR = 29.23
        #[0.1, 'PnP_ML_Moreau_INIT', {'lambda':2.25, 'g_param':0.0813, 'stepsz_coeff':2.25, }],  # PSNR = 28.47
        #[0.1, 'PnP_SCUNet', {'stepsz_coeff':1.83, }],  # PSNR = 24.87
        #[0.1, 'PnP_ML_SCUNet_init', {'stepsz_coeff':2.0, }],  # PSNR = 25.01
        #[0.1, 'PnP_prox', {'g_param': 0.28, 'stepsz_coeff': 2.0, }],  # PSNR = 26.54
        #[0.1, 'PnP_prox_ML_INIT', {'g_param': 0.1, 'stepsz_coeff': 1.62, }],  # PSNR = 29.11
    ]
    if ConfParam().use_equivariance is True:
        gs_vec = [
            [0.1, 'FB_TV_ML', {'lambda':0.0833, }],  # PSNR = 22.19
            [0.1, 'PnP', {'g_param':0.16, 'stepsz_coeff':2.0, }],  # PSNR = 28.19
            [0.1, 'PnP_INIT', {'g_param':0.14, 'stepsz_coeff':2.5, }],  # PSNR = 29.42
            [0.1, 'PnP_ML', {'g_param':0.14, 'stepsz_coeff':1.67, }],  # PSNR = 28.00
            [0.1, 'PnP_ML_INIT', {'g_param':0.0801, 'stepsz_coeff':1.17, }],  # PSNR = 29.23
            [0.1, 'PnP_ML_Moreau', {'lambda':1.5, 'g_param':0.105, 'stepsz_coeff':3.0, }],  # PSNR = 27.27
            [0.1, 'PnP_ML_Moreau_INIT', {'lambda':2.44, 'g_param':0.0813, 'stepsz_coeff':2.63, }],  # PSNR = 28.47
            [0.1, 'PnP_prox', {'g_param':0.26, 'stepsz_coeff':2.0, }],  # PSNR = 26.47
            [0.1, 'PnP_prox_ML_INIT', {'g_param':0.1, 'stepsz_coeff':1.62, }],  # PSNR = 29.14
            [0.1, 'PnP_SCUNet', {'stepsz_coeff':2.0, }],  # PSNR = 25.07
            [0.1, 'PnP_ML_SCUNet_init', {'stepsz_coeff':2.17, }],  # PSNR = 25.05
        ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)

def mri_hyper_param(noise_pow, gs_key):
    pass

def tomography_hyper_param(noise_pow, gs_key):
    pass
