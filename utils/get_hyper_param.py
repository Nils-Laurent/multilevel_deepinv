
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
        [0.1, 'PnP_ML_INIT', {'lambda': 0.22, 'g_param': 0.0501, }],  # PSNR = 19.61
        [0.1, 'FB_TV_ML', {'lambda': 0.04, }],  # (0.04 fine)
        [0.1, 'PnP_prox_ML_INIT', {'g_param': 0.25, }],  # PSNR = 16.58
    ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)

def poisson_hyper_param(noise_pow, gs_key):
    gs_vec = [
        [0.1, 'PnP_ML_INIT', {'lambda': 0.22, 'g_param': 0.07, }],
        [0.1, 'PnP_prox_ML_INIT', {'g_param': 0.25, }],
        [0.1, 'FB_TV_ML', {'lambda': 1.0, }],
    ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)


def inpainting_hyper_param(noise_pow, gs_key):
    gs_vec = [
        [0.1, 'PnP_ML_INIT', {'lambda': 1e-05, 'g_param': 0.0701, }],  # PSNR = 26.56
        [0.1, 'PnP_prox_ML_INIT', {'lambda': 1e-05, 'g_param': 0.0701, }],
        [0.1, 'FB_TV_ML', {'lambda': 0.12, }],  # PSNR = 22.03
    ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)

def mri_hyper_param(noise_pow, gs_key):
    gs_vec = [
        [0.1, 'PnP_ML_INIT', {'lambda': 1e-05, 'g_param': 0.0751, }],  # PSNR = 28.92
        [0.1, 'FB_TV_ML', {'lambda': 0.15, }],  # PSNR = 25.31
    ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)

def tomography_hyper_param(noise_pow, gs_key):
    pass
