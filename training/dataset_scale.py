import os

import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from multilevel.info_transfer import BlackmannHarris
from training.transforms_scale import ToCoarserScale
from utils.paths import dataset_path


def gen_scale_dataset(dataset_name, scale_count):
    if scale_count <= 0:
        print('scale count must be positive')
        return None

    DATA_DIR = dataset_path()

    comp_vec = []
    for i in range(scale_count):
        comp_vec.append(ToCoarserScale(def_filter=BlackmannHarris()))
    tr = transforms.Compose(comp_vec)

    tr_tensor = transforms.ToTensor()

    root_path = DATA_DIR / dataset_name
    dataset = torchvision.datasets.ImageFolder(root=root_path, transform=tr_tensor)
    scaled_name = dataset_name + f"_coarse{scale_count}"

    save_dir = DATA_DIR / scaled_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"generating images (scale_count = {scale_count})")
    for x_tuple, m_tuple in zip(dataset, dataset.imgs):
        x = x_tuple[0]
        x_coarse = tr(x)
        cur_path = m_tuple[0]

        suffix = os.path.relpath(cur_path, root_path)
        new_path = save_dir / suffix
        new_path.parents[0].mkdir(parents=True, exist_ok=True)
        save_image(x_coarse, new_path)

