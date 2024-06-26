
def select_param(noise_pow, noise_vec, p_red, p_tv):
    for n in range(len(noise_vec)):
        if noise_vec[n] == noise_pow:
            p_red = p_red[n]
            p_tv = p_tv[n]
            break

    if type(p_red) is not dict:
        raise NotImplementedError()

    return p_red, p_tv

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
    return select_param(noise_pow, noise_vec, p_red, p_tv)

def blur_hyper_param(noise_pow):
    noise_vec = [0.01, 0.05, 0.1, 0.2, 0.3]

    p_red = [
        {'lambda': 0.0003/noise_pow**2, 'g_param': 0.0713}, # not gridsearh
        {'lambda': 0.0013/noise_pow**2, 'g_param': 0.0913},
        {'lambda': 0.0040/noise_pow**2, 'g_param': 0.1280},
        {'lambda': 0.0126/noise_pow**2, 'g_param': 0.1144},
        {'lambda': 0.0300/noise_pow**2, 'g_param': 0.1102},
    ]

    p_tv = [
        {'lambda': 0.0047}, # not gridsearh
        {'lambda': 0.0147},
        {'lambda': 0.0596},
        {'lambda': 0.1509},
        {'lambda': 0.2545},
    ]
    return select_param(noise_pow, noise_vec, p_red, p_tv)

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

    return select_param(noise_pow, noise_vec, p_red, p_tv)
