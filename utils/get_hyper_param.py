
def inpainting_hyper_param(noise_pow):
    noise_vec = [0.05, 0.1, 0.2, 0.3]
    if noise_pow == noise_vec[0]:
        p_red  = {'lambda': 0.0050, 'g_param': 0.0921}
        p_tv  = {'lambda': 0.0477}
    elif noise_pow == noise_vec[1]:
        p_red  = {'lambda': 0.0100, 'g_param': 0.0904}
        p_tv  = {'lambda': 0.1357}
    elif noise_pow == noise_vec[2]:
        p_red  = {'lambda': 0.0200, 'g_param': 0.0913}
        p_tv  = {'lambda': 0.3053}
    elif noise_pow == noise_vec[3]:
        p_red  = {'lambda': 0.0300, 'g_param': 0.0904}
        p_tv  = {'lambda': 0.5150}
    else:
        raise NotImplementedError()

    return p_red, p_tv

def blur_hyper_param(noise_pow):
    noise_vec = [0.05, 0.1, 0.2, 0.3]
    if noise_pow == noise_vec[0]:
        p_red  = {'lambda': 0.0013, 'g_param': 0.0913}
        p_tv  = {'lambda': 0.0147}
    elif noise_pow == noise_vec[1]:
        p_red  = {'lambda': 0.0040, 'g_param': 0.1280}
        p_tv  = {'lambda': 0.0596}
    elif noise_pow == noise_vec[2]:
        p_red  = {'lambda': 0.0126, 'g_param': 0.1144}
        p_tv  = {'lambda': 0.1509}
    elif noise_pow == noise_vec[3]:
        p_red  = {'lambda': 0.0300, 'g_param': 0.1102}
        p_tv  = {'lambda': 0.2545}
    else:
        raise NotImplementedError()

    return p_red, p_tv

def tomography_hyper_param(noise_pow):
    noise_vec = [0.05, 0.1, 0.2, 0.3]
    if noise_pow == noise_vec[0]:
        p_red  = {'lambda': 7.9245e-05, 'g_param': 0.0904}
        p_tv  = {'lambda': 0.0007}
    elif noise_pow == noise_vec[1]:
        p_red  = {'lambda': 0.0002, 'g_param': 0.0913}
        p_tv  = {'lambda': 0.0014}
    elif noise_pow == noise_vec[2]:
        p_red  = {'lambda': 0.0003, 'g_param': 0.0904}
        p_tv  = {'lambda': 0.0044}
    elif noise_pow == noise_vec[3]:
        p_red  = {'lambda': 0.0005, 'g_param': 0.0904}
        p_tv  = {'lambda': 0.0067}
    else:
        raise NotImplementedError()

    return p_red, p_tv
