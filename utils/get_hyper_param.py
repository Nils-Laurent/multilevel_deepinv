
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

def inpainting_hyper_param(noise_pow):
    noise_vec = [0.01, 0.05, 0.1, 0.2, 0.3]
    p_red = [
        {'lambda': 0.0020/noise_pow**2, 'g_param': 0.0921}, # not gridsearh
        {'lambda': 0.0050/noise_pow**2, 'g_param': 0.0921},
        {'lambda': 0.0100/noise_pow**2, 'g_param': 0.0904},
        {'lambda': 0.0200/noise_pow**2, 'g_param': 0.0913},
        {'lambda': 0.0300/noise_pow**2, 'g_param': 0.0904},
    ]
    p_tv = [
        {'lambda': 0.0177}, # not gridsearh
        {'lambda': 0.0477},
        {'lambda': 0.1357},
        {'lambda': 0.3053},
        {'lambda': 0.5150},
    ]
    return select_param(noise_pow, noise_vec, p_red, p_pnp, p_tv)

def blur_hyper_param(noise_pow):
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
        {'lambda': 0.0047}, # not gridsearh
        {'lambda': 0.0147},
        {'lambda': 0.0596},
        {'lambda': 0.1509},
        {'lambda': 0.2545},
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

    p_tv = [
        {'lambda': 0.0007},
        {'lambda': 0.0014},
        {'lambda': 0.0044},
        {'lambda': 0.0067},
    ]

    return select_param(noise_pow, noise_vec, p_red, p_pnp, p_tv)
