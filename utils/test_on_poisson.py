import deepinv
from deepinv.loss import PSNR
from deepinv.models import DRUNet
from deepinv.optim import optim_builder, Zero, PnP
from deepinv.optim.optim_iterators import GDIteration, PGDIteration
from deepinv.physics import PoissonNoise, Denoising
from deepinv.utils.demo import load_dataset
from torchvision import transforms

from multilevel_utils.custom_poisson_noise import CPoissonNoise, CPoissonLikelihood
from utils.paths import dataset_path


def poisson_test(device):
    original_data_dir = dataset_path()
    t_crop = (1024, 1024)
    transform_vec = [transforms.CenterCrop(t_crop), transforms.ToTensor(), ]
    val_transform = transforms.Compose(transform_vec)
    dataset = load_dataset("cset", original_data_dir, transform=val_transform)
    x_ref = dataset[0][0].unsqueeze(0).to(device)

    bkg = 1
    gain = 1/30
    #n_model = CPoissonNoise(gain=gain, bkg=bkg)
    n_model = CPoissonNoise(gain=gain)

    physics = Denoising(noise_model=n_model, device=device)
    datafidelity = CPoissonLikelihood(gain=gain, bkg=bkg, denormalize=True)

    pl_cst = 1 / (gain * bkg)**2  # Lipschitz cst
    stepsize = 1 / pl_cst
    #stepsize = 0.01
    g_param = 0.05
    print("stepsize =", stepsize, 'g_param =', g_param)

    params_algo = {"stepsize": stepsize, "g_param": g_param}

    d1 = DRUNet(pretrained="download", device=device)
    d2 = deepinv.models.EquivariantDenoiser(d1, random=True)

    model = optim_builder(
        iteration=PGDIteration(),
        prior=PnP(denoiser=d1),
        data_fidelity=datafidelity,
        max_iter=10,
        g_first=False,
        early_stop=True,
        crit_conv='residual',
        thres_conv=1e-6,
        verbose=True,
        params_algo=params_algo,
        #custom_init=f_init,
    )

    model.eval()
    y = physics(x_ref)  # poisson noise
    x_est, met = model(y, physics, x_gt=x_ref, compute_metrics=True)
    print("psnr :", met['psnr'][0])

    return None