import numpy


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
    #gs_vec = [
    #    #{'lambda': 0.00022175929916556925, 'g_param': 0.028326647356152534, },
    #    [0.01, 'PnP_ML_prox', {'g_param': 0.0285, }],  # PSNR = 21.32
    #    [0.01, 'FB_TV_ML', {'lambda': 0.00124, }],  # PSNR = 21.19
    #    [0.01, 'RED_ML_INIT', {'lambda': 0.0275, 'g_param': 0.139, }],  # PSNR = 21.11
    #    [0.1, 'PnP_ML_prox', {'g_param': 0.0583, }],  # PSNR = 20.70
    #    [0.1, 'FB_TV_ML', {'lambda': 0.0275, }],  # PSNR = 19.15
    #    [0.1, 'RED_ML_INIT', {'lambda': 0.154, 'g_param': 0.139, }],  # PSNR = 20.34
    #    [0.2, 'PnP_ML_prox', {'g_param': 0.104, }],  # PSNR = 19.80
    #    [0.2, 'FB_TV_ML', {'lambda': 0.0773, }],  # PSNR = 18.10
    #    [0.2, 'RED_ML_INIT', {'lambda': 0.306, 'g_param': 0.185, }],  # PSNR = 19.59
    #]

    gs_vec = [
        [0.01, 'PnP_ML_prox', {'g_param': 0.0401, }],  # PSNR = 20.86
        [0.01, 'PnP_ML', {'lambda': 1e-05, 'g_param': 0.0751, }],  # PSNR = 20.06
        [0.01, 'FB_TV_ML', {'lambda': 1e-05, }],  # PSNR = 20.96
        [0.01, 'RED_ML_INIT', {'lambda': 0.04, 'g_param': 0.0951, }],  # PSNR = 21.09
        [0.1, 'PnP_ML_prox', {'g_param': 0.0451, }],  # PSNR = 20.56
        [0.1, 'PnP_ML', {'lambda': 0.01, 'g_param': 0.0801, }],  # PSNR = 19.86
        [0.1, 'FB_TV_ML', {'lambda': 0.02, }],  # PSNR = 19.12
        [0.1, 'RED_ML_INIT', {'lambda': 0.12, 'g_param': 0.155, }],  # PSNR = 20.37
        [0.2, 'PnP_ML_prox', {'g_param': 0.0751, }],  # PSNR = 19.77
        [0.2, 'PnP_ML', {'lambda': 1e-05, 'g_param': 0.1, }],  # PSNR = 19.42
        [0.2, 'FB_TV_ML', {'lambda': 0.08, }],  # PSNR = 18.09
        [0.2, 'RED_ML_INIT', {'lambda': 0.5, 'g_param': 0.13, }],  # PSNR = 19.65
    ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)

def inpainting_hyper_param(noise_pow, gs_key):
    #noise_vec = [0.01, 0.05, 0.1, 0.2, 0.3]
    #p_red = [
    #    {'lambda': 0.0020/noise_pow**2, 'g_param': 0.0921}, # not gridsearh
    #    {'lambda': 0.0050/noise_pow**2, 'g_param': 0.0921},
    #    {'lambda': 0.0100/noise_pow**2, 'g_param': 0.0904},
    #    {'lambda': 0.0200/noise_pow**2, 'g_param': 0.0913},
    #    {'lambda': 0.0300/noise_pow**2, 'g_param': 0.0904},
    #]
    #p_pnp = [
    #    {'lambda': 0.0020/noise_pow**2, 'g_param': 0.0921}, # not gridsearh
    #    {'lambda': 0.0050/noise_pow**2, 'g_param': 0.0921},
    #    {'lambda': 0.0100/noise_pow**2, 'g_param': 0.0904},
    #    {'lambda': 0.0200/noise_pow**2, 'g_param': 0.0913},
    #    {'lambda': 0.0300/noise_pow**2, 'g_param': 0.0904},
    #]
    #p_tv = [
    #    {'lambda': 0.0177}, # not gridsearh
    #    {'lambda': 0.0477},
    #    {'lambda': 0.1357},
    #    {'lambda': 0.3053},
    #    {'lambda': 0.5150},
    #]
    #return select_param(noise_pow, noise_vec, p_red, p_pnp, p_tv)
    gs_vec = [
        [0.01, 'PnP_ML_prox', {'g_param': 0.15, }],  # PSNR = 25.53
        [0.01, 'PnP_ML', {'lambda': 1e-05, 'g_param': 0.0801, }],  # PSNR = 29.21
        [0.01, 'FB_TV_ML', {'lambda': 0.02, }],  # PSNR = 27.21
        [0.01, 'RED_ML_INIT', {'lambda': 0.5, 'g_param': 0.105, }],  # PSNR = 29.02
        [0.1, 'PnP_ML_prox', {'g_param': 0.125, }],  # PSNR = 26.06
        [0.1, 'PnP_ML', {'lambda': 1e-05, 'g_param': 0.0801, }],  # PSNR = 28.75
        [0.1, 'FB_TV_ML', {'lambda': 0.12, }],  # PSNR = 23.33
        [0.1, 'RED_ML_INIT', {'lambda': 0.99, 'g_param': 0.0851, }],  # PSNR = 25.57
        [0.2, 'PnP_ML_prox', {'g_param': 0.13, }],  # PSNR = 25.56
        [0.2, 'PnP_ML', {'lambda': 1e-05, 'g_param': 0.14, }],  # PSNR = 26.16
        [0.2, 'FB_TV_ML', {'lambda': 0.3, }],  # PSNR = 20.73
        [0.2, 'RED_ML_INIT', {'lambda': 1.0, 'g_param': 0.115, }],  # PSNR = 21.47
    ]

    res = gs_pick_bounds(gs_vec, gs_key=gs_key, noise_pow=noise_pow)
    if isinstance(res, dict):
        return res
    else:
        return affine_interpolation(res, noise_pow=noise_pow)

def backup_blur_hyper_param(noise_pow):
    noise_vec = [0.01, 0.05, 0.1, 0.2, 0.3]

    p_red = [
        #blur_0.01_RED_ML_INIT_scatter2d PSNR = 21.10517692565918
        {'lambda': 0.027509065344929695, 'g_param': 0.13888375461101532},
        {'lambda': 0.0013/noise_pow**2, 'g_param': 0.0913}, # not gridsearch
        #blur_0.1_RED_ML_INIT_scatter2d PSNR = 20.32094383239746
        {'lambda': 0.15388204157352448, 'g_param': 0.1201939731836319, },
        #blur_0.2_RED_ML_INIT_scatter2d PSNR = 19.555994033813477
        {'lambda': 0.4323298931121826, 'g_param': 0.13888373970985413, },
        {'lambda': 0.0300/noise_pow**2, 'g_param': 0.1102}, # not gridsearch
    ]

    p_pnp = [
        #blur_0.01_PnP_ML_scatter2d PSNR = 21.24113655090332
        {'lambda': 0.00022175929916556925, 'g_param': 0.028326647356152534, },
        {'lambda': 0.00022175929916556925, 'g_param': 0.028326647356152534, },
        #blur_0; .1; _PnP_ML_scatter2d; PSNR = 20.695531845092773
        {'lambda': 0.0012404919834807515, 'g_param': 0.05834972858428955, },
        #blur_0; .2; _PnP_ML_scatter2d; PSNR = 19.82026481628418
        {'lambda': 0.0017503963317722082, 'g_param': 0.10401930660009384, },
        {'lambda': 0.0017503963317722082, 'g_param': 0.10401930660009384, },
    ]

    p_tv = [
    #    {'lambda': 0.0047}, # not gridsearh
    #    {'lambda': 0.0147},
    #    {'lambda': 0.0596},
    #    {'lambda': 0.1509},
    #    {'lambda': 0.2545},

        #blur_0.01_FB_TV_ML_plot1d PSNR = 21.19106101989746
        {'lambda': 0.0012404919834807515, },

        {'lambda': 0.004, }, # not gridsearch

        #blur_0.1_FB_TV_ML_plot1d PSNR = 19.14695167541504
        {'lambda': 0.027509065344929695, },

        #blur_0.2_FB_TV_ML_plot1d PSNR = 18.103586196899414
        {'lambda': 0.07728640735149384, },

        {'lambda': 0.14, }, # not gridsearch
    ]

    return select_param(noise_pow, noise_vec, p_red, p_pnp, p_tv)

def tomography_hyper_param(noise_pow):
    noise_vec = [0.05, 0.1, 0.2, 0.3]

    p_red = [
        {'lambda': 7.9245e-05/noise_pow**2, 'g_param': 0.0904},
        {'lambda': 0.0002/noise_pow**2, 'g_param': 0.0913},
        {'lambda': 0.0003/noise_pow**2, 'g_param': 0.0904},
        {'lambda': 0.0005/noise_pow**2, 'g_param': 0.0904},
    ]

    p_pnp = [
        {'lambda': 7.9245e-05/noise_pow**2, 'g_param': 0.0904},
        {'lambda': 0.0002/noise_pow**2, 'g_param': 0.0913},
        {'lambda': 0.0003/noise_pow**2, 'g_param': 0.0904},
        {'lambda': 0.0005/noise_pow**2, 'g_param': 0.0904},
    ]

    p_tv = [
        {'lambda': 0.0007},
        {'lambda': 0.0014},
        {'lambda': 0.0044},
        {'lambda': 0.0067},
    ]

    return select_param(noise_pow, noise_vec, p_red, p_pnp, p_tv)
