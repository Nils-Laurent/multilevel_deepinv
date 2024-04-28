import deepinv
import torch
import matplotlib.pyplot as plt

def plot_fft_dataset(dataset):
    original_data_dir = dataset_path()
    img_size = 256
    val_transform = transforms.Compose(
        [transforms.CenterCrop(img_size), transforms.ToTensor()]
    )
    dataset = load_dataset('set3c', original_data_dir, transform=val_transform)
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    id_img = 0
    for t in dataset:
        id_img += 1
        img = t[0].unsqueeze(0).to(device)
        plot_fft_mod(img)

def plot_fft_mod(img):
    img_sum = torch.sum(img, dim=1, keepdim=True)
    f_img = torch.fft.fft2(img_sum, dim=(2, 3))
    f_img = torch.fft.fftshift(f_img, dim=(2, 3))
    m_img = torch.abs(f_img)

    #deepinv.utils.plot(img)
    #deepinv.utils.plot(m_img)
    n_img = torch.log(1 + m_img)
    y_min = n_img.min()
    y_max = n_img.max()
    plt.imshow(n_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), vmin=y_min, vmax=y_max, cmap="gray")
    plt.colorbar()
    plt.show()