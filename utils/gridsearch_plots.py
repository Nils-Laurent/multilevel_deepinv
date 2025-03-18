import numpy
from os.path import join
from matplotlib import pyplot
from utils.paths import get_out_dir
import torch

import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
from matplotlib import rcParams
rcParams['text.latex.preamble'] = r'\newcommand{\mathdefault}[1][]{}'

def tune_scatter_2d(d_tune, keys, fig_name=None):
    v_min = numpy.min(list(map(lambda el: torch.min(el['cost']), d_tune)))
    v_max = numpy.max(list(map(lambda el: torch.max(el['cost']), d_tune)))

    pyplot.figure()
    s = 4.0

    x = []
    y = []
    z = []
    for rec in range(len(d_tune)):
        coord = d_tune[rec]['coord']
        cost = d_tune[rec]['cost']
        #s *= 0.5
        for id_xy in numpy.ndindex(cost.shape):
            x.append(coord[0][id_xy[0]])
            y.append(coord[1][id_xy[1]])
            z.append(cost[id_xy])

    pyplot.scatter(x, y, c=z, s=s, cmap='copper')
    #pyplot.scatter(x, y, c=z, s=s, cmap='copper', norm=matplotlib.colors.LogNorm())

    #pyplot.xscale('log')
    #pyplot.yscale('log')
    pyplot.xlabel(keys[0])
    pyplot.ylabel(keys[1])
    pyplot.colorbar()
    pyplot.show()
    if not fig_name is None:
        out_path = get_out_dir()
        #pyplot.savefig(join(out_path, (fig_name + ".pgf")))
        pyplot.savefig(join(out_path, (fig_name + ".png")))
    pyplot.close('all')


def tune_plot_1d(d_tune, keys, fig_name=None):
    pyplot.figure()
    for rec in range(len(d_tune)):
        coord = d_tune[rec]['coord']
        cost = d_tune[rec]['cost']

        x = []
        y = []
        for id_xy in numpy.ndindex(cost.shape):
            x.append(coord[0][id_xy[0]])
            y.append(cost[id_xy])
        pyplot.plot(x, y)

    #pyplot.xscale('log')
    pyplot.xlabel(keys[0])
    pyplot.show()
    if not fig_name is None:
        out_path = get_out_dir()
        #pyplot.savefig(join(out_path, (fig_name + ".pgf")))
        pyplot.savefig(join(out_path, (fig_name + ".png")))
    pyplot.close('all')

def print_gridsearch_max(key_, d_tune, keys, noise_pow):

    psnr_tensor = d_tune[-1]['cost']
    coord_vec = d_tune[-1]['coord']

    max_v = -torch.inf
    for n in range(0, len(d_tune)):
        psnr_tmp = d_tune[n]['cost']
        ftens = psnr_tmp[~torch.isnan(psnr_tmp)]

        if ftens.numel() == 0:
            break
        if torch.max(ftens) < max_v:
            break

        psnr_tensor = torch.nan_to_num(d_tune[n]['cost'], -torch.inf)
        max_v = torch.max(psnr_tensor.view(-1))
        coord_vec = d_tune[n]['coord']
        max_j = torch.argmax(psnr_tensor.view(-1))
        max_i = torch.unravel_index(max_j, psnr_tensor.shape)

    print(f"[{noise_pow}, '{key_}', ", end="")
    print("{", end="")
    it = 0
    for k in keys:
        val = coord_vec[it][max_i[it]]
        print(f"'{k}':" + f"{val:.3}" + ", ", end="")
        it += 1
    print("}]," + "  # PSNR = " + "{:.2f}".format(max_v))
