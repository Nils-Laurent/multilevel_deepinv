import matplotlib.pyplot as plt
import numpy

import torch
from deepinv.models import DRUNet
from deepinv.physics import GaussianNoise
from torch.utils.data import Subset
from torchvision import transforms

import deepinv
from deepinv.utils.demo import load_dataset
from utils.paths import dataset_path

def image_fft(img):
    img_sum = torch.sum(img, dim=1, keepdim=True)
    f_img = torch.fft.fft2(img_sum, dim=(2, 3))

    # fftshift: origin is at image center
    f_img = torch.fft.fftshift(f_img, dim=(2, 3))

    return f_img

def img_energy_ratio(img):
    fe_img = torch.abs(image_fft(img))

    Lx = img.shape[2]
    Ly = img.shape[3]
    x0 = round(img.shape[2]/4)
    y0 = round(img.shape[3]/4)

    e_low_f = torch.sum(fe_img[:, :, x0:(Lx - x0), y0:(Ly - y0)])
    e_img = torch.sum(fe_img)

    return e_low_f/e_img

def img_domain_ratio(img, e_ratio):
    fft_modulus = torch.abs(image_fft(img))

    lx = img.shape[2]
    ly = img.shape[3]

    k_max = min(lx, ly)

    s_ratio = 1.0
    for k in range(round(k_max/2), 0, -1):
        e_low_f = torch.sum(fft_modulus[:, :, k:(lx - k), k:(ly - k)])
        e_img = torch.sum(fft_modulus)
        ratio = e_low_f/e_img

        if ratio >= e_ratio:
            s_ratio = (k_max - 2 * k) / k_max
            break

    return s_ratio


def sigma_domain_ratio(img, denoiser, sigma_vec):
    y_vec = []
    for sigma in sigma_vec:
        noise_model = GaussianNoise(sigma=sigma)
        denoised_img = denoiser(noise_model(img), sigma)
        y = img_domain_ratio(denoised_img, 0.9)
        y_vec.append(y)

    return y_vec


def sigma_psd_ratio(img, denoiser, sigma_vec):
    y_vec = []
    for sigma in sigma_vec:
        d_img = denoiser(img, sigma)
        y = img_energy_ratio(d_img)
        y_vec.append(y.item())

    return y_vec


def plot_spectr_ratio():
    #sigma_vec = numpy.logspace(numpy.log10(0.01), numpy.log10(0.4), 100)
    sigma_vec = numpy.logspace(numpy.log10(0.01), numpy.log10(1.0), 100)
    nb_img = 500 # should be at least 500 => 25 curves below/above 5%/95% quantile
    #nb_img = 20
    bp_thresh = 20

    original_data_dir = dataset_path()
    img_size = 178
    val_transform = transforms.Compose(
        [transforms.CenterCrop(img_size), transforms.ToTensor()]
    )
    #dataset = load_dataset('set3c', original_data_dir, transform=val_transform)
    dataset = load_dataset('celeba', original_data_dir, transform=val_transform)
    dataset = Subset(dataset, range(0, nb_img))
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    denoiser = DRUNet(pretrained="download", train=False, device=device)

    id_img = 0
    res_data = numpy.zeros((nb_img, len(sigma_vec)))
    for t in dataset:
        id_img += 1
        print(f"processing img f{id_img}")
        img = t[0].unsqueeze(0).to(device)
        #y_vec = sigma_psd_ratio(img, denoiser, sigma_vec)
        y_vec = sigma_domain_ratio(img, denoiser, sigma_vec)
        if nb_img <= bp_thresh:
            plt.plot(sigma_vec, y_vec)
            continue

        res_data[id_img - 1, :] = y_vec

    if nb_img <= bp_thresh:
        plt.xlabel("sigma")
        plt.ylabel("range")
        plt.show()
        return

    y5 = numpy.quantile(res_data, q=0.05, axis=0)
    y25 = numpy.quantile(res_data, q=0.25, axis=0)
    y50 = numpy.quantile(res_data, q=0.5, axis=0)
    y75 = numpy.quantile(res_data, q=0.75, axis=0)
    y95 = numpy.quantile(res_data, q=0.95, axis=0)

    y_mean = numpy.mean(res_data, axis=0)

    fig, ax = plt.subplots()
    ax.fill_between(sigma_vec, y5, y95, alpha=0.1, color='goldenrod')
    ax.fill_between(sigma_vec, y25, y75, alpha=0.1, color='darkorchid')
    ax.plot(sigma_vec, y50, color='darkorchid')
    ax.plot(sigma_vec, y_mean, color='gray')
    plt.xlabel("sigma")
    plt.ylabel("range")
    plt.show()

def plot_img_spectr_magnitude(img):
    m_img = torch.abs(image_fft(img))

    n_img = torch.log(1 + m_img)
    y_min = n_img.min()
    y_max = n_img.max()
    plt.imshow(n_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), vmin=y_min, vmax=y_max, cmap="gray")
    plt.colorbar()
    plt.show()