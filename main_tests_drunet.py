import os
import deepinv
import numpy
import torch
import torch.utils.benchmark as benchmark
from deepinv.models import DRUNet
from deepinv.physics import GaussianNoise
from deepinv.utils.demo import load_dataset
from torchvision import transforms

from tests.test_lipschitz import measure_lipschitz
from utils.paths import dataset_path, get_out_dir


def measure_exec_time():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(device)

    val_transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    original_data_dir = dataset_path()
    dataset = load_dataset("celeba_hq", original_data_dir, transform=val_transform)

    img = None
    for t in dataset:
        img = t[0].unsqueeze(0).to(device)
        break

    print(img.shape)

    sigma = torch.tensor(0.08).to(device)
    denoiser = DRUNet(pretrained="download", device=device)
    noise_model = GaussianNoise(sigma=sigma)
    img_n = noise_model(img, device=device)
    y_time = []

    # warmup
    buff = denoiser(img_n, sigma)
    buff = denoiser(img_n, sigma)

    vec_size = [32, 64, 128, 256, 512, 1024]
    nb_pixels = [px**2 for px in vec_size]
    for k in vec_size:
        img_it = transforms.CenterCrop(k)(img_n)
        t = benchmark.Timer(
            stmt="denoiser(img_it, sigma)",
            globals={"img_it": img_it, "denoiser": denoiser, "sigma": sigma}
        )
        y_time.append(t.timeit(number=16).mean)

    f_name = os.path.join(get_out_dir(), "data_drunet_time.npy")
    numpy.save(f_name, [vec_size, nb_pixels, y_time])

    #deepinv.utils.plot_curves(metrics={"time": [y_time]})
    #fig = pyplot.figure()
    #pyplot.plot(list(vec_size), y_time)
    #pyplot.xlabel("image size (pixels)")
    #pyplot.ylabel("denoiser time (seconds)")
    #pyplot.show()

    #tikzplotlib.save("DRUNet_time.tex", fig=fig)


def main_lipschitz():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    denoiser = DRUNet(device=device)
    sigma_vec = [0.02 + n * 0.001 for n in range(0, 200)]

    # measure_lipschitz(denoiser, sigma_vec=sigma_vec, device=device, sigma_noise=0.1)
    # measure_lipschitz(denoiser, sigma_vec=sigma_vec, device=device, sigma_noise=0.2)

    # sigma_noise is None => denoiser match the true noise level
    measure_lipschitz(denoiser, sigma_vec=sigma_vec, device=device, sigma_noise=None)

if __name__ == "__main__":
    # plot_spectr_ratio()
    # main_lipschitz()
    measure_exec_time()