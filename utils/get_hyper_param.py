
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
    gs_vec = [
        [0.1, 'PnP_ML_INIT', {'g_param':0.0601, }],  # PSNR = 20.03
        [0.1, 'PnP_ML_Moreau_INIT', {'lambda':1.06, 'g_param':0.0601, }],  # PSNR = 20.07
        [0.1, 'FB_TV_ML', {'lambda':0.0278, }],  # PSNR = 19.01
        [0.1, 'PnP_ML_DnCNN_Moreau_init', {'lambda':0.417, }],  # PSNR = 18.50
        [0.1, 'PnP_ML_SCUNet_Moreau_init', {'lambda':1.56, }],  # PSNR = 11.41
        [0.1, 'PnP_prox_ML_INIT', {'g_param':0.144, }],  # PSNR = 20.11
        [0.1, 'PnP_prox_ML_Moreau_INIT', {'lambda':0.0833, 'g_param':0.0661, }],  # PSNR = 20.80
        #[0.1, 'PnP_ML_INIT', {'g_param': 0.0541, }],  # PSNR = 19.99
        #[0.1, 'PnP_ML_Moreau_INIT', {'lambda': 0.8, 'g_param': 0.0541, }],  # PSNR = 20.09
        #[0.1, 'FB_TV_ML', {'lambda': 0.04, }],  # PSNR = 18.96
        #[0.1, 'PnP_prox_ML_INIT', {'g_param': 0.216, }],  # PSNR = 19.52
    ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)

def poisson_hyper_param(noise_pow, gs_key):
    gs_vec = [
        [0.1, 'PnP_ML_INIT', {'g_param': 0.05, }],
        [0.1, 'PnP_ML_Moreau_INIT', {'lambda': 0.22, 'g_param': 0.05, }],
        [0.1, 'PnP_prox_ML_INIT', {'g_param': 0.05, }],
        [0.1, 'FB_TV_ML', {'lambda': 1.0, }],
    ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)


def inpainting_hyper_param(noise_pow, gs_key):
    gs_vec = [
        [0.1, 'PnP_ML_INIT', {'g_param':0.0661, }],  # PSNR = 29.20
        [0.1, 'PnP_ML_Moreau_INIT', {'lambda':1.28, 'g_param':0.0661, }],  # PSNR = 29.18
        [0.1, 'FB_TV_ML', {'lambda':0.139, }],  # PSNR = 23.46
        [0.1, 'PnP_ML_DnCNN_Moreau_init', {'lambda':1.69, }],  # PSNR = 19.88
        [0.1, 'PnP_ML_SCUNet_Moreau_init', {'lambda':1.0, }],  # PSNR = 24.54
        [0.1, 'PnP_prox_ML_INIT', {'g_param':0.108, }],  # PSNR = 29.06
        [0.1, 'PnP_prox_ML_Moreau_INIT', {'lambda':1.72, 'g_param':0.108, }],  # PSNR = 29.13
        #[0.1, 'PnP_ML_INIT', {'g_param': 0.0701, }],  # PSNR = 29.19
        #[0.1, 'PnP_ML_Moreau_INIT', {'lambda': 0.38, 'g_param': 0.0701, }],  # PSNR = 29.22
        #[0.1, 'FB_TV_ML', {'lambda': 0.14, }],  # PSNR = 23.44
        #[0.1, 'PnP_prox_ML_INIT', {'g_param': 0.11, }],  # PSNR = 29.05
    ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)

def demosaicing_hyper_param(noise_pow, gs_key):
    gs_vec = [
        [0.1, 'PnP_ML_INIT', {'g_param':0.0601, }],  # PSNR = 29.08
        [0.1, 'PnP_ML_Moreau_INIT', {'lambda':0.361, 'g_param':0.0601, }],  # PSNR = 29.12
        [0.1, 'FB_TV_ML', {'lambda':0.0833, }],  # PSNR = 22.19
        [0.1, 'PnP_ML_DnCNN_Moreau_init', {'lambda':0.167, }],  # PSNR = 20.30
        [0.1, 'PnP_ML_SCUNet_Moreau_init', {'lambda':0.333, }],  # PSNR = 21.12
        [0.1, 'PnP_prox_ML_INIT', {'g_param':0.0961, }],  # PSNR = 29.11
        [0.1, 'PnP_prox_ML_Moreau_INIT', {'lambda':0.5, 'g_param':0.102, }],  # PSNR = 29.18
        #[0.1, 'PnP_ML_INIT', {'g_param': 0.0601, }],  # PSNR = 29.06
        #[0.1, 'PnP_ML_Moreau_INIT', {'lambda': 0.2, 'g_param': 0.0601, }],  # PSNR = 29.17
        #[0.1, 'FB_TV_ML', {'lambda': 0.1, }],  # PSNR = 22.19
        #[0.1, 'PnP_prox_ML_INIT', {'g_param': 0.175, }],  # PSNR = 27.51
    ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)

def mri_hyper_param(noise_pow, gs_key):
    gs_vec = [
        [0.1, 'PnP_ML_INIT', {'g_param': 0.0701, }],  # PSNR = 30.97
        [0.1, 'PnP_ML_Moreau_INIT', {'lambda': 0.84, 'g_param': 0.0701, }],  # PSNR = 31.08
        [0.1, 'FB_TV_ML', {'lambda': 0.18, }],  # PSNR = 25.57
    ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)

def tomography_hyper_param(noise_pow, gs_key):
    pass
