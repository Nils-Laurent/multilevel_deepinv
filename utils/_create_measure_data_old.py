import glob
import sys

import torch
from torch.utils.data import Subset
from torchvision import transforms
from itertools import product

from gen_fig.fig_metric_logger import MRedMLInit, MRedInit, MRed, MRedML, MDPIR, MFb, MFbMLGD, MPnPML, MPnP

if "/.fork" in sys.prefix:
    sys.path.append('/projects/UDIP/nils_src/deepinv')

import deepinv
from deepinv.physics import GaussianNoise
from deepinv.utils.demo import load_dataset
from tests.parameters import get_parameters_tv, get_parameters_red, get_parameters_pnp_prox, single_level_params
from tests.test_alg import RunAlgorithm
from tests.utils import physics_from_exp, data_from_user_input, ResultManager
from utils.npy_utils import save_grid_tune_info, load_variables_from_npy, grid_search_npy_filename
from utils.gridsearch import tune_grid_all
from utils.gridsearch_plots import tune_scatter_2d, tune_plot_1d, print_gridsearch_max
from utils.paths import dataset_path, get_out_dir, measurements_path


def create_measure_data_(
        problem,
        noise_pow=0.001,
        dataset_name='set3c',
        img_size=None,
        test_data=True,
):
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    print(device)

    original_data_dir = dataset_path()
    if img_size is None:
        val_transform = transforms.Compose([transforms.ToTensor()])
    else:
        val_transform = transforms.Compose([transforms.CenterCrop(img_size), transforms.ToTensor()])

    dataset = load_dataset(dataset_name, original_data_dir, transform=val_transform)
    dataset = Subset(dataset, range(1))

    # inpainting: proportion of pixels to keep
    params_exp = {'problem': problem, 'set_name': dataset_name, 'shape': (3, img_size, img_size)}
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
    _, problem_name = physics_from_exp(params_exp, g, device)

    measurement_dir = measurements_path()
    degrad_name = dataset_name + "_" + problem_name
    m_data_dir = measurement_dir / degrad_name

    dtest = None
    dtrain = None
    if test_data:
        dtest = dataset
    else:
        dtrain = dataset

    num_workers = 1

    get_ph = lambda tupl: tupl[0]
    physics_list = [get_ph(physics_from_exp(params_exp, g, device)) for i in range(len(dataset))]

    deepinv.datasets.generate_dataset(
        train_dataset=None,
        test_dataset=dataset,
        batch_size=1,
        physics=physics_list,
        device=device,
        save_dir=m_data_dir,
        num_workers=num_workers,
    )

    nfiles = 0
    for file in glob.glob(m_data_dir.__str__() + "/*.h5"):
        nfiles += 1

    for i in range(nfiles):
        fi = m_data_dir / ("dinv_dataset"+str(i)+".h5")
        state_path = m_data_dir / ("physics"+str(i)+".pt")
        state_dict = torch.load(state_path)
        ph_test, pname = physics_from_exp(params_exp, g, device)
        ph_test.load_state_dict(state_dict)
        d2 = deepinv.datasets.HDF5Dataset(path=fi, train=False)
        x, y = d2[0]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        deepinv.utils.plot([x, y])
        deepinv.utils.plot([ph_test.A(x)])
        deepinv.utils.plot([y])
