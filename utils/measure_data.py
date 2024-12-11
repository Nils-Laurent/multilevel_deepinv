import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

if "/.fork" in sys.prefix:
    sys.path.append('/projects/UDIP/nils_src/deepinv')

from torch.utils.data import Subset, Dataset
import deepinv
from deepinv.physics import GaussianNoise
from deepinv.utils.demo import load_dataset
from utils.utils import physics_from_exp
from utils.paths import dataset_path, measurements_path


def create_measure_data(
        problem,
        noise_pow,
        dataset_name,
        img_size=None,
):
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(device)

    original_data_dir = dataset_path()
    val_transform = transforms.Compose([transforms.ToTensor()])
    if not(img_size is None) and type(img_size) is int:
        val_transform = transforms.Compose([transforms.CenterCrop(img_size), transforms.ToTensor()])
        img_size = (3, img_size, img_size)

    # inpainting: proportion of pixels to keep
    params_exp = {'problem': problem, 'set_name': dataset_name, 'shape': img_size}
    params_exp['noise_pow'] = noise_pow
    if problem == 'inpainting':
        params_exp[problem] = 0.5
    elif problem == 'tomography':
        params_exp[problem] = 0.6
    elif problem == 'blur':
        params_exp[problem + '_pow'] = 3.6
    else:
        raise NotImplementedError()

    tensor_np = torch.tensor(noise_pow).to(device)
    g = GaussianNoise(sigma=tensor_np)

    dataset = load_dataset(dataset_name, original_data_dir, transform=val_transform)
    gen_physics = lambda: physics_from_exp(params_exp, g, device)[0]
    physics_list = [gen_physics() for i in range(len(dataset))]
    physics_state = [ph.state_dict() for ph in physics_list]
    degrad_list = []
    for xt, ph in zip(dataset, physics_list):
        x = xt[0].to(device)
        degrad_list.append((x, ph(x).squeeze(0)))
    data = {'tuples': degrad_list, 'states': physics_state}


    unused_ph, problem_name = physics_from_exp(params_exp, g, device)
    measure_name = dataset_name + "_" + problem_name

    measurements_dir = measurements_path()
    degrad_dir = measurements_dir / measure_name
    os.makedirs(degrad_dir, exist_ok=True)

    data_file = degrad_dir / (measure_name + ".pth")
    torch.save(data, data_file)


def load_measure_data(params_exp, device, subset_size=None, target=None):
    #device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    #print(device)

    # inpainting: proportion of pixels to keep
    noise_pow = params_exp['noise_pow']
    dataset_name = params_exp['set_name']

    tensor_np = torch.tensor(noise_pow).to(device)
    g = GaussianNoise(sigma=tensor_np)
    gen_physics = lambda: physics_from_exp(params_exp, g, device)[0]

    ph0, problem_name = physics_from_exp(params_exp, g, device)
    measurement_dir = measurements_path()
    degrad_name = dataset_name + "_" + problem_name
    file_name = measurement_dir / degrad_name / (degrad_name + ".pth")

    file_data = torch.load(file_name)
    state_list2 = file_data['states']
    degrad2 = file_data['tuples']

    physics_list = []
    for item in range(len(state_list2)):
        ph = gen_physics()
        ph.load_state_dict(state_list2[item])
        physics_list.append(ph)

    dataset = DegradDataset(degrad2)

    if (not (subset_size is None)) or (not (target is None)):
        if subset_size is None:
            subset_size = 1

        offset = target
        if target is None:
            offset = 0

        dataset = Subset(dataset, range(offset, offset + subset_size))
        physics_list = physics_list[offset:offset + subset_size]

    res_dataloader = DataLoader(dataset)

    return res_dataloader, physics_list


class DegradDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
