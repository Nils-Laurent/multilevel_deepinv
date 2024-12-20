import deepinv
import torch
from torchvision import transforms
from deepinv.optim import Zero, L2
from deepinv.physics import GaussianNoise, BlurFFT
from deepinv.physics.blur import gaussian_blur
from deepinv.utils.demo import load_dataset

from multilevel.iterator import CoarseModel, MultiLevelParams
from multilevel.info_transfer import CFir
from utils.parameters import standard_multilevel_param
from utils.paths import dataset_path


def params():
    # We assume regularization gradient is 1-Lipschitz
    params_algo = {
        #'cit': SincFilter(),
        'cit': CFir(),
        'scale_coherent_grad': True
    }

    p_tv = params_algo.copy()

    lambda_tv = 0.5
    print("lambda_tv:", lambda_tv)

    iters_fine = 200
    iters_coarse = 5
    iters_vec = [iters_coarse, iters_coarse, iters_coarse, iters_fine]
    iters_vec = [iters_coarse, iters_fine]
    p_tv['iml_max_iter'] = 3

    p_tv = standard_multilevel_param(p_tv, it_vec=iters_vec, lambda_fine=lambda_tv)
    p_tv['lip_g'] = 1.0
    p_tv['prox_crit'] = 1e-6
    p_tv['prox_max_it'] = 1000
    gamma_vec = [1.1] * len(iters_vec)
    gamma_vec[-1] = 1.0
    lambda_vec = p_tv['params_multilevel'][0]['lambda']
    lf = 1.0  # data-fidelity gradient lipschitz cst
    step_coeff = 1.9  # convex setting
    stepsize_vec = [step_coeff / (lf + 1.0/gamma) for gamma in gamma_vec]
    stepsize_vec[-1] = step_coeff / (lf + lambda_vec[-1])
    p_tv = _finalize_params(p_tv, lambda_vec, stepsize_vec, gamma_vec)

    return p_tv

def test_blur():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    noise_pow = 0
    tensor_np = torch.tensor(noise_pow).to(device)
    noise_model = GaussianNoise(sigma=tensor_np)

    width = 1024
    shape = (3, width, width)
    power = 3.6
    physics = BlurFFT(img_size=shape, noise_model=noise_model,
                      filter=gaussian_blur(sigma=(power, power), angle=0), device=device)

    prior = Zero()
    l2 = L2()

    ml_params = params()
    ml_params_cl = MultiLevelParams(ml_params)

    cm = CoarseModel(prior, l2, physics, ml_params_cl)

    original_data_dir = dataset_path()
    val_transform = transforms.Compose([transforms.CenterCrop(width), transforms.ToTensor()])
    dataset = load_dataset('astro_ml', original_data_dir, transform=val_transform)
    x0 = dataset[0][0].to(device).unsqueeze(0)

    #x0_p = cm.projection(x0)
    #print("x0_p min = ", torch.min(x0_p))
    #print("x0_p max = ", torch.max(x0_p))
    #print("x0_p med = ", torch.median(x0_p))
    #ds_op = cit.op
    #nn = ds_op.compute_norm(x0)
    #print(nn)

    y = physics(x0)
    y_coarse = cm.projection(y)

    cit = cm.cit_op
    x_coarse = cit.projection(x0)
    v1 = cit.projection(physics.A_adjoint(physics.A(cit.prolongation(x_coarse))))
    v2 = cm.physics.A_adjoint(cm.physics.A(x_coarse))

    print("difference : L_infty = ", torch.max(torch.abs(v1.reshape(-1) - v2.reshape(-1))))

    v1n = v1 / torch.max(v1)
    v2n = v2 / torch.max(v2)
    deepinv.utils.plot([v1n - y_coarse, v2n - y_coarse])
    deepinv.utils.plot([v1 - y_coarse, v2 - y_coarse])

    diff_n = torch.abs(v1n - v2n)
    deepinv.utils.plot([diff_n])
    print("difference : normalized L_infty = ", torch.max(torch.abs(v1n.reshape(-1) - v2n.reshape(-1))))

if __name__ == '__main__':
    test_blur()