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

    pyplot.xscale('log')
    pyplot.yscale('log')
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

    pyplot.xscale('log')
    pyplot.xlabel(keys[0])
    pyplot.show()
    if not fig_name is None:
        out_path = get_out_dir()
        #pyplot.savefig(join(out_path, (fig_name + ".pgf")))
        pyplot.savefig(join(out_path, (fig_name + ".png")))
    pyplot.close('all')