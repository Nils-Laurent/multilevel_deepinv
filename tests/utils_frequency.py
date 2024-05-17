import matplotlib.pyplot as plt

import torch
from deepinv.models import DRUNet
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
        d_img = denoiser(img, sigma)
        y = img_domain_ratio(d_img, 0.9)
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
    sigma_vec = [0.02 + n * 0.01 for n in range(0, 98)]


    original_data_dir = dataset_path()
    img_size = 256
    val_transform = transforms.Compose(
        [transforms.CenterCrop(img_size), transforms.ToTensor()]
    )
    dataset = load_dataset('set3c', original_data_dir, transform=val_transform)
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    denoiser = DRUNet(pretrained="download", train=False, device=device)

    id_img = 0
    for t in dataset:
        id_img += 1
        img = t[0].unsqueeze(0).to(device)
        #y_vec = sigma_psd_ratio(img, denoiser, sigma_vec)
        y_vec = sigma_domain_ratio(img, denoiser, sigma_vec)
        plt.plot(sigma_vec, y_vec)

    plt.show()


def plot_img_spectr_magnitude(img):
    m_img = torch.abs(image_fft(img))

    n_img = torch.log(1 + m_img)
    y_min = n_img.min()
    y_max = n_img.max()
    plt.imshow(n_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), vmin=y_min, vmax=y_max, cmap="gray")
    plt.colorbar()
    plt.show()