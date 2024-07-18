import os
from os.path import join

import numpy

import numpy
from matplotlib import pyplot
from pylatex import Document, TikZ, Axis, Plot, NoEscape, Command

from utils.paths import get_out_dir


def fig_net_time_imgsz(exp_name):
    f_name = os.path.join(get_out_dir(), f"{exp_name}.npy")
    np_res = numpy.load(f_name)
    [vec_sz, nb_pixels, y_time] = np_res
    doc = Document()
    with (doc.create(TikZ())):
        ax_options = f'height=10cm, width=16cm, grid=major, xlabel=nb_pixels, ylabel=time, legend pos=south east'
        with doc.create(Axis(options=ax_options)) as ax:
            y_coord = [(d, t) for (d, t) in zip(nb_pixels, y_time)]
            ax.append(Plot(coordinates=y_coord))

    with (doc.create(TikZ())):
        ax_options = f'height=10cm, width=16cm, grid=major, xlabel=img_size, ylabel=time, legend pos=south east'
        with doc.create(Axis(options=ax_options)) as ax:
            y_coord = [(d, t) for (d, t) in zip(vec_sz, y_time)]
            ax.append(Plot(coordinates=y_coord))

    out_f = join(get_out_dir(), exp_name).__str__()
    doc.generate_tex(filepath=out_f)


def fig_net_time_depth(exp_name):
    f_name = os.path.join(get_out_dir(), f"{exp_name}.npy")
    np_res = numpy.load(f_name)
    [vec_depth, y_time] = np_res
    doc = Document()
    with (doc.create(TikZ())):
        ax_options = f'height=10cm, width=16cm, grid=major, xlabel=depth, ylabel="time (ms)", legend pos=south east'
        with doc.create(Axis(options=ax_options)) as ax:
            y_coord = [(d, t) for (d, t) in zip(vec_depth, y_time)]
            ax.append(Plot(coordinates=y_coord))

    out_f = join(get_out_dir(), exp_name).__str__()
    doc.generate_tex(filepath=out_f)


if __name__ == "__main__":
    np_res = numpy.load('out/data_drunet_time.npy')
    [nb_pixels, vec_size, y_time] = np_res
    fig = pyplot.figure()
    pyplot.plot(list(vec_size), y_time, linestyle='--', marker='o')
    pyplot.xlabel("image size (pixels)")
    pyplot.ylabel("denoiser time (seconds)")
    pyplot.show()