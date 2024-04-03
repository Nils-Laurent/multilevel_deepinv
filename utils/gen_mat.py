from pathlib import Path
from scipy.io import savemat
from os.path import join
import utils.paths as paths
import torch


def dict_str(dict_in):
    if 'cit' in dict_in:
        dict_in['cit'] = str(dict_in['cit'])
    return str(dict_in)


def log_param_in_dict(dict_in, dict_params):
    d2 = {'params_algo': {}}

    for k_ in dict_params.keys():
        if not isinstance(dict_params[k_], dict):
            d2['params_algo'][k_] = dict_params[k_]
        elif k_ == 'cit':
            d2[k_] = str(dict_params[k_])
        else:
            d2[k_] = dict_params[k_]

    for k_ in d2.keys():
        if k_ in dict_in:
            print(f"warning: cannot log params of {k_}")
        else:
            dict_in[k_] = dict_str(d2[k_])


def gen_mat_cost(dict_cost, f_name, dict_params):
    r"""
    :param dict dict_cost: contains costs for all algorithms
    :param str f_name: file name
    :param dict_params:
    """

    # if cost is list of scalar tensor, convert to normal scalar
    for k_ in dict_cost.keys():
        if len(dict_cost[k_]) == 0:
            continue

        if isinstance(dict_cost[k_][0], torch.Tensor):
            dict_cost[k_] = [x.item() for x in dict_cost[k_]]

    log_param_in_dict(dict_cost, dict_params)

    f_name_ext = f_name + "_costs" + ".mat"
    out_f = join(paths.get_out_dir(), f_name_ext)
    savemat(out_f, dict_cost)


def img_np_convention(x):
    x = x.squeeze()
    # check if it is a rgb image
    if len(x.shape) > 2:
        x = x.permute(1, 2, 0)
    try:
        x = x.cpu().numpy()
    except:
        x = x.cpu().detach().numpy()
    return x


def gen_mat_images(dict_images, f_name, dict_params):
    # if image is a tensor, convert to normal matrix
    for k_ in dict_images.keys():
        if isinstance(dict_images[k_], torch.Tensor):
            dict_images[k_] = img_np_convention(dict_images[k_])

    log_param_in_dict(dict_images, dict_params)

    f_name_ext = f_name + "_images" + ".mat"
    out_f = join(paths.get_out_dir(), f_name_ext)
    savemat(out_f, dict_images)


def gen_mat_y_data(dict_y_data, f_name, dict_params):
    # if image is a tensor, convert to normal matrix
    for k_ in dict_y_data.keys():
        if isinstance(dict_y_data[k_], torch.Tensor):
            dict_y_data[k_] = img_np_convention(dict_y_data[k_])

    log_param_in_dict(dict_y_data, dict_params)

    f_name_ext = f_name + "_y_data" + ".mat"
    out_f = join(paths.get_out_dir(), f_name_ext)
    savemat(out_f, dict_y_data)


def gen_matlab_conf(exp):
    f_name_ext = "param_last.m"
    out_f = join(paths.get_out_dir(), f_name_ext)
    Path(out_f).touch()
    f = open(out_f, "r+")
    f.truncate(0)
    f.write('exp_name = "' + exp + '";\n')
    f.close()


def gen_mat(data, f_name):
    out_f = join(paths.get_out_dir(), f_name)
    savemat(out_f, data)
