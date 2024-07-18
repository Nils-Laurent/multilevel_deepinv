import os
import deepinv
import numpy
import torch
from deepinv.physics import GaussianNoise
from deepinv.utils.demo import load_dataset
from torchvision import transforms
from utils.paths import dataset_path, get_out_dir

def measure_net_time_depth(f_net, net_name, device):
    vec_depth = list(range(5, 25))
    img_sz = 512
    n_rep = 100

    original_data_dir = dataset_path()
    val_transform = transforms.Compose([transforms.ToTensor()])
    dataset = load_dataset("celeba_hq", original_data_dir, transform=val_transform)
    for t in dataset:
        img = t[0].unsqueeze(0).to(device)
        break
    img_it = transforms.CenterCrop(img_sz)(img)

    sigma = torch.tensor(0.08).to(device)
    y_time = []

    time_iter = True
    start = torch.cuda.Event(enable_timing=time_iter)
    end = torch.cuda.Event(enable_timing=time_iter)
    for k in vec_depth:
        net = f_net(k)

        # warmup
        start.record()
        net(img_it, sigma)
        net(img_it, sigma)
        end.record()
        torch.cuda.synchronize()
        start.elapsed_time(end)
        # warmup

        it_time = 0  # Time reported in milliseconds
        for jk in range(n_rep):
            start.record()
            net(img_it, sigma)
            end.record()
            torch.cuda.synchronize()
            it_time += start.elapsed_time(end) # Time reported in milliseconds
        it_time = it_time / n_rep  # Time reported in milliseconds

        y_time.append(it_time)

    exp_name = f"{net_name}_time_depth"
    f_name = os.path.join(get_out_dir(), f"{exp_name}.npy")
    print(f"write {f_name}")
    numpy.save(f_name, [vec_depth, y_time])
    return exp_name


def measure_net_time_imgsize(net, net_name, device):
    val_transform = transforms.Compose([transforms.ToTensor()])
    original_data_dir = dataset_path()
    dataset = load_dataset("celeba_hq", original_data_dir, transform=val_transform)

    img = None
    for t in dataset:
        img = t[0].unsqueeze(0).to(device)
        break

    print(img.shape)

    sigma = torch.tensor(0.08).to(device)
    noise_model = GaussianNoise(sigma=sigma)
    img_n = noise_model(img, device=device)
    y_time = []

    vec_size = [32, 64, 128, 256, 512, 1024]
    nb_pixels = [px**2 for px in vec_size]
    time_iter = True
    start = torch.cuda.Event(enable_timing=time_iter)
    end = torch.cuda.Event(enable_timing=time_iter)
    n_rep = 100
    for k in vec_size:
        img_it = transforms.CenterCrop(k)(img_n)

        if k == 0:  # warmup
            start.record()
            net(img_it, sigma)
            net(img_it, sigma)
            end.record()
            net(img_it, sigma)
            net(img_it, sigma)
            torch.cuda.synchronize()
            start.elapsed_time(end)  # Time reported in milliseconds

        it_time = 0
        for jk in range(n_rep):
            start.record()
            net(img_it, sigma)
            end.record()
            torch.cuda.synchronize()
            it_time += start.elapsed_time(end)  # Time reported in milliseconds
        it_time = it_time / n_rep  # Time reported in milliseconds

        y_time.append(it_time)

    exp_name = f"{net_name}_time_imgsz"
    f_name = os.path.join(get_out_dir(), f"{exp_name}.npy")
    print(f"write {f_name}")
    numpy.save(f_name, [vec_size, nb_pixels, y_time])
    return exp_name