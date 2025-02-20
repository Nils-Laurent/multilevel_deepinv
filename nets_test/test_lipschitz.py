from pathlib import Path
import torch
import deepinv
from deepinv.loss import JacobianSpectralNorm
from deepinv.physics import GaussianNoise
from deepinv.utils.demo import load_dataset
from torchvision import transforms

from torch.utils.data import DataLoader
from deepinv import train

from utils.mat_utils import gen_mat
from utils.paths import dataset_path, measurements_path, checkpoint_path
from torchvision import datasets


def measure_lipschitz(denoiser, sigma_vec, device, sigma_noise):
    img_size = 128 if torch.cuda.is_available() else 32

    val_transform = transforms.Compose(
        [transforms.CenterCrop(img_size), transforms.ToTensor()]
    )
    # add batch and channel dimensions
    dataset_name = 'set3c'
    ORIGINAL_DATA_DIR = Path("../tests") / "datasets"
    dataset_tuples = load_dataset(dataset_name, ORIGINAL_DATA_DIR, transform=val_transform)
    dataset = []
    for t in dataset_tuples:
        dataset.append(t[0].unsqueeze(0).to(device))

    # val_transform = transforms.CenterCrop(img_size)
    # dataset = get_astro3(dtype=torch.FloatTensor, transform=val_transform, device=device)

    for x0 in dataset:
        deepinv.utils.plot(x0)

    reg_l2 = JacobianSpectralNorm(max_iter=10, tol=1e-3, eval_mode=False)
    if sigma_noise is not None:
        g0 = GaussianNoise(sigma=sigma_noise)
    else:
        g0 = None
    lipschitz = []
    for sigma in sigma_vec:
        sigma_data = {'sigma': sigma, 'cst': None}
        lip_vec = []
        print("sigma =", sigma)
        for x0 in dataset:
            if g0 is None:
                g = GaussianNoise(sigma=sigma)
            else:
                g = g0

            y = g(x0)
            x = y.clone().requires_grad_()
            # out = 1/(sigma**2) * (denoiser(x, sigma=sigma) - x)
            out = denoiser(x, sigma=sigma) - x
            regval = reg_l2(out, x)
            lip_vec.append(regval.item())
            print(regval.item())
        sigma_data['cst'] = lip_vec
        lipschitz.append(sigma_data)

    gen_mat({'lipschitz': lipschitz}, f'lipschitz_{sigma_noise}.mat')

