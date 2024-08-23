from os.path import join

import numpy
from matplotlib import pyplot
from pylatex import Document, TikZ, Axis, Plot, NoEscape, Command
import pylatex

from tests.parameters import get_parameters_pnp, get_parameters_pnp_prox_noreg, get_parameters_red, \
    get_parameters_tv_coarse_pgd, get_parameters_tv, get_parameters_pnp_approx_noreg, get_parameters_pnp_prox, \
    get_parameters_pnp_approx, get_parameters_pnp_prox_nc, get_parameters_pnp_approx_nc, get_parameter_pnp_Moreau
from utils.paths import get_out_dir
from dataclasses import dataclass

@dataclass
class MPnP:
    key = 'PnP'
    color = 'purple'
    linestyle = 'dashed'
    label = key
    param_fn = get_parameters_pnp_prox
@dataclass
class MPnPMoreau:
    key = 'PnP_Moreau'
    color = 'purple'
    linestyle = 'dashed'
    label = key
    param_fn = get_parameter_pnp_Moreau
@dataclass
class MPnPMLApproxNc:
    key = 'PnP_ML_approx_N1c'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_pnp_approx_nc
@dataclass
class MPnPMLNc:
    key = 'PnP_ML_N1c'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_pnp_prox_nc
@dataclass
class MPnPMLApprox:
    key = 'PnP_ML_approx'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_pnp_approx
@dataclass
class MPnPMLProx:
    key = 'PnP_ML_prox'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_pnp_prox
@dataclass
class MPnPMLApproxNoR:
    key = 'PnP_ML_approx_NoReg'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_pnp_approx_noreg
@dataclass
class MPnPMLNoR:
    key = 'PnP_ML_NoReg'
    color = 'purple'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_pnp_prox_noreg
@dataclass
class MPnPML:
    key = 'PnP_ML'
    color = 'gray'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_pnp
@dataclass
class MRed:
    key = 'RED'
    color = 'red'
    linestyle = 'dashed'
    label = key
    param_fn = get_parameters_red
@dataclass
class MRedML:
    key = 'RED_ML'
    color = 'red'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_red
@dataclass
class MRedInit:
    key = 'RED_INIT'
    color = 'black'
    linestyle = 'dashed'
    label = key
    param_fn = get_parameters_red
@dataclass
class MRedMLInit:
    key = 'RED_ML_INIT'
    color = 'black'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_red
@dataclass
class MDPIR:
    key = 'DPIR'
    color = 'green'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_red
@dataclass
class MFb:
    key = 'FB_TV'
    color = 'blue'
    linestyle = 'dashed'
    label = key
    param_fn = get_parameters_tv
@dataclass
class MFbMLGD:
    key = 'FB_TV_ML'
    color = 'blue'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_tv
@dataclass
class MFbMLProx:
    key = 'FB_TV_ML_prox'
    color = 'blue'
    linestyle = 'solid'
    label = key
    param_fn = get_parameters_tv_coarse_pgd

class GenFigMetricLogger:
    def __init__(self):
        self.data = {}
        self.keep_vec = []

    def add_logger(self, x, mkey):
        self.data[mkey] = x

    def set_keep_vec(self, def_vec):
        self.keep_vec = def_vec

    def _gen_tex_compute_means(self, x_vec, mat):
        y = numpy.mean(mat, axis=0)
        c = [(xi, yi) for (xi, yi) in zip(x_vec, y)]
        y_std = numpy.std(mat, axis=0)
        y_low = y - y_std
        y_up = y + y_std
        c_low = [(xi, yi) for (xi, yi) in zip(x_vec, y_low)]
        c_up = [(xi, yi) for (xi, yi) in zip(x_vec, y_up)]

        d_coord = {"c_low": c_low, "c_up": c_up, "c": c}

        return d_coord

    def _gen_tex_compute_quantiles(self, x_vec, mat):
        y_low = numpy.quantile(mat, q=0.05, axis=0)
        c_low = [(xi, yi) for (xi, yi) in zip(x_vec, y_low)]
        y_up = numpy.quantile(mat, q=0.95, axis=0)
        c_up = [(xi, yi) for (xi, yi) in zip(x_vec, y_up)]
        y = numpy.median(mat, axis=0)
        c = [(xi, yi) for (xi, yi) in zip(x_vec, y)]

        d_coord = {"c_low": c_low, "c_up": c_up, "c": c}

        return d_coord

    def _gen_tex_coord(self, x_vec, mat, option):
        if option == "median":
            return self._gen_tex_compute_quantiles(x_vec, mat)

        return self._gen_tex_compute_means(x_vec, mat)

    def gen_tex(self, metric, fig_name, coord_opt, x_axis=None):
        doc = Document()
        doc.preamble.append(Command('usepgfplotslibrary', 'fillbetween'))
        x_label = x_axis
        if x_axis is None:
            x_label = 'iterations'

        with (doc.create(TikZ())):
            ax_options = f'height=10cm, width=16cm, grid=major, xlabel={x_label}, ylabel=PSNR, legend pos=south east'
            with doc.create(Axis(options=ax_options)) as ax:
                index = 0
                for method_key, g in self.data.items():
                    if not g.has_key(metric) or not self.is_valid_key(method_key):
                        continue
                    if not (method_key in self.keep_vec.keys()):
                        continue
                    index += 1

                    mat = g.metric_matrix(metric)
                    method = self.methods[method_key]
                    mx, my = mat.shape
                    if x_axis is None:
                        x_vec = range(my)
                    else:
                        x_mat = g.metric_matrix(x_axis)
                        x_vec = numpy.median(x_mat, axis=0)  # robust to operating system events
                        x_vec = numpy.cumsum(x_vec) / 1000  # ms to seconds

                    d_coord = self._gen_tex_coord(x_vec, mat, coord_opt)

                    ref1 = f"A{index}"
                    ref2 = f"B{index}"
                    opt1 = f"mark=none, forget plot, opacity=0, name path={ref1}"
                    opt2 = f"mark=none, forget plot, opacity=0, name path={ref2}"

                    ax.append(Plot(coordinates=d_coord['c_low'], options=opt1))
                    ax.append(Plot(coordinates=d_coord['c_up'], options=opt2))
                    fill_opts = f'{method.color}, forget plot, opacity=0.1'
                    fill_tex = r'\addplot' + f'[{fill_opts}] fill between[of={ref1} and {ref2}];'
                    ax.append(NoEscape(fill_tex))

                    plot_options = f"mark=none, color={method.color}, {method.linestyle}"
                    ax.append(Plot(name=method.label, coordinates=d_coord['c'], options=plot_options))

        full_name = fig_name + "_" + metric + "_" + coord_opt
        if not(x_axis is None):
            full_name += "_" + x_axis
        out_f = join(get_out_dir(), full_name).__str__()
        doc.generate_tex(filepath=out_f)

    #def gen_fig(self, metric):
    #    fig, ax = pyplot.subplots()

    #    for method_key, g in self.data.items():
    #        if not g.has_key(metric) or not self.is_valid_key(method_key):
    #            continue

    #        mat = g.metric_matrix(metric)
    #        method = self.methods[method_key]
    #        mx, my = mat.shape
    #        y5 = numpy.quantile(mat, q=0.05, axis=0)
    #        y95 = numpy.quantile(mat, q=0.95, axis=0)
    #        y_med = numpy.median(mat, axis=0)
    #        ax.fill_between(range(my), y5, y95, alpha=0.15, color=method.color)
    #        pyplot.plot(range(my), y_med, color=method.color, linestyle=method.linestyle, label=method.label)
    #        #ax.fill_between(range(my), y5, y95, alpha=0.1, color='goldenrod')
    #        #ax.fill_between(sigma_vec, y25, y75, alpha=0.1, color='darkorchid')
    #    pyplot.xlabel("x")
    #    pyplot.ylabel("y")
    #    pyplot.legend(loc="lower right")
    #    pyplot.show()

    #    out_path = get_out_dir()
    #    pyplot.savefig(join(out_path, (metric + ".png")))
    #    pyplot.close('all')
