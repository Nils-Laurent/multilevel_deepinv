import numpy
import torch
from scipy.io import savemat

from utils.gridsearch_plots import tune_scatter_2d, tune_plot_1d, print_gridsearch_max
from utils.npy_utils import grid_search_npy_filename, load_variables_from_npy
from itertools import product


def gen_mat_data(pb_list, in_noise_pow_vec):
    noise_pow_vec = numpy.sort(in_noise_pow_vec)
    for pb, noise_pow in product(pb_list, noise_pow_vec):
        print(f"pb = {pb}")
        file_pb = grid_search_npy_filename(suffix=pb + str(noise_pow))
        data = load_variables_from_npy(file_pb)

        for key_ in data.keys():
            if key_ == 'PnP' or key_ == 'PnP_ML_INIT':
                axis = data[key_]['axis']
                tensors = data[key_]['tensors']

                gridsearch_to_mat(key_, tensors, axis, pb, noise_pow)


def gridsearch_to_mat(key_, d_tune, keys, pb, noise_pow):
    psnr_tensor = d_tune[0]['cost']
    coord_vec = d_tune[0]['coord']

    fname = f"gs_{key_}_{pb}_{noise_pow}.mat"

    mat_data = {'tensor': psnr_tensor.numpy()}
    for i in range(0, len(keys)):
        mat_data[keys[i]] = coord_vec[i].numpy()
    savemat(fname, mat_data)


def main_gridsearch_mat():
    vec_noise_pow = [0.1]
    vec_pb = ['demosaicing']
    gen_mat_data(vec_pb, vec_noise_pow)
    vec_pb = ['inpainting']
    gen_mat_data(vec_pb, vec_noise_pow)
    vec_pb = ['blur']
    gen_mat_data(vec_pb, vec_noise_pow)


if __name__ == '__main__':
    main_gridsearch_mat()
