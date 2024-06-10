import torch
from torch.utils.data import DataLoader, Dataset

from deepinv.physics import Inpainting, Blur
from deepinv.physics.blur import gaussian_blur, BlurFFT
from deepinv.datasets import HDF5Dataset
from torchvision import transforms

from multilevel_utils.radon import Tomography


def physics_from_exp(params_exp, noise_model, device):
    noise_pow = params_exp['noise_pow']
    problem = params_exp['problem']

    match problem:
        case 'inpainting':
            def_mask = params_exp[problem]
            print("def_mask:", def_mask)
            problem_full = problem + "_" + str(def_mask) + "_" + str(noise_pow)
            physics = Inpainting(params_exp['shape'], mask=def_mask, noise_model=noise_model, device=device)
        case 'tomography':
            prop = params_exp[problem]
            def_angles = int(180*prop)
            print("angle prop:", prop)
            problem_full = problem + "_" + str(prop) + "_" + str(noise_pow)
            physics = Tomography(
                angles=def_angles, img_width=params_exp['shape'][-2], noise_model=noise_model, device=device
            )
        case 'blur':
            power = params_exp[problem + '_pow']
            print("def_blur_pow:", power)
            problem_full = problem + "_" + str(power) + "_" + str(noise_pow)
            #physics = Blur(gaussian_blur(sigma=(power, power), angle=0), noise_model=noise_model, device=device, padding='replicate')
            physics = BlurFFT(img_size=params_exp['shape'], filter=gaussian_blur(sigma=(power, power), angle=0), device=device)
        case _:
            raise NotImplementedError("Problem " + problem + " not supported")

    return physics, problem_full


class CH5Dataset(HDF5Dataset):
    def __init__(self, *args, img_size = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tr = None
        if img_size is not None:
            self.tr = transforms.CenterCrop(img_size)

    def __getitem__(self, item):
        x, y = super().__getitem__(item)
        if self.tr is not None:
            x = self.tr(x)
            y = self.tr(y)
        return x, y


def data_from_user_input(input_data, physics, params_exp, problem_name, device):
    if isinstance(input_data, torch.Tensor):
        data = input_data
    elif isinstance(input_data, Dataset):
        return DataLoader(input_data, shuffle=True)
    else:
        raise NotImplementedError()
        #save_dir = measurements_path().joinpath(params_exp['set_name'], problem_name)
        #f_prefix = str(save_dir.joinpath('**', '*.'))
        #find = ""
        #find_file = ""
        #for filename in glob.iglob(f_prefix + 'h5', recursive=True):
        #    print(filename)
        #    find = "h5"
        #    find_file = filename

        #match find:
        #    case 'h5':
        #        data_bis = CH5Dataset(img_size=params_exp["shape"][1:2], path=find_file, train=False)
        #    case _:
        #        # create dataset if it does not exist
        #        generate_dataset(
        #            train_dataset=input_data, physics=physics, save_dir=save_dir, device=device, test_dataset=input_data
        #        )
        #        data_bis = CH5Dataset(img_size=params_exp["shape"][1:2], path=find_file, train=False)
        #data = DataLoader(data_bis, shuffle=True, batch_size=1)
        ##data = DataLoader(data_bis, shuffle=False)

    return data


def single_level_params(params_ml):
    params = params_ml.copy()
    params['n_levels'] = 1
    params['level'] = 1
    params['iters'] = params_ml['params_multilevel'][0]['iters'][-1]

    return params

def standard_multilevel_param(params, it_vec):
    levels = len(it_vec)
    ml_dict = {"iters": it_vec}
    params['params_multilevel'] = [ml_dict]
    params['level'] = levels
    params['n_levels'] = levels

    iml_max_iter = params['iml_max_iter']
    params['multilevel_step'] = [k < iml_max_iter for k in range(0, it_vec[-1])]

    return params


from prettytable import PrettyTable
def count_parameters(model, pr=True, namenet=''):
    table = PrettyTable(["Modules", "Parameters"])

    total_params = 0
    total_params_dict = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
        total_params_dict[name] = parameter

    if pr == True:
        print(table)

    print(namenet + " has total Trainable Params: ", total_params)

    return total_params, total_params_dict

