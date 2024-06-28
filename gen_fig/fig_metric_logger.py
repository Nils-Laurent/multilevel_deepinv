from os.path import join

import numpy
from matplotlib import pyplot
from pylatex import Document, TikZ, Axis, Plot, NoEscape, Command
import pylatex

from utils.paths import get_out_dir
from dataclasses import dataclass

@dataclass
class MPnP:
    key = 'PnP'
    color = 'purple'
    linestyle = 'dashed'
    label = key
@dataclass
class MPnPML:
    key = 'PnP_ML'
    color = 'purple'
    linestyle = 'solid'
    label = key
@dataclass
class MRed:
    key = 'RED'
    color = 'red'
    linestyle = 'dashed'
    label = key
@dataclass
class MRedML:
    key = 'RED_ML'
    color = 'red'
    linestyle = 'solid'
    label = key
@dataclass
class MRedInit:
    key = 'RED_INIT'
    color = 'black'
    linestyle = 'dashed'
    label = key
@dataclass
class MRedMLInit:
    key = 'RED_ML_INIT'
    color = 'black'
    linestyle = 'solid'
    label = key
@dataclass
class MDPIR:
    key = 'DPIR'
    color = 'green'
    linestyle = 'solid'
    label = key
@dataclass
class MFb:
    key = 'FB_TV'
    color = 'blue'
    linestyle = 'dashed'
    label = key
@dataclass
class MFbML:
    key = 'FB_TV_ML'
    color = 'blue'
    linestyle = 'solid'
    label = key

def methods_obj():
    t_res = [
        MRed(),
        MRedML(),
        MRedInit(),
        MRedMLInit(),
        MDPIR(),
        MFb(),
        MFbML(),
    ]

    res = {}
    for el in t_res:
        res[el.key] = el

    return res


class GenFigMetricLogger:
    def __init__(self):
        self.data = {}
        self.methods = methods_obj()

    def add_logger(self, x, mkey):
        self.data[mkey] = x

    def is_valid_key(self, mkey):
        return mkey in self.methods.keys()

    def gen_tex(self, metric, x_axis=None):
        doc = Document()
        doc.preamble.append(Command('usepgfplotslibrary', 'fillbetween'))

        with (doc.create(TikZ())):
            ax_options = 'height=10cm, width=16cm, grid=major, xlabel=iterations, ylabel=PSNR, legend pos=south east'
            with doc.create(Axis(options=ax_options)) as ax:
                index = 0
                for method_key, g in self.data.items():
                    if not g.has_key(metric) or not self.is_valid_key(method_key):
                        continue
                    index += 1

                    mat = g.metric_matrix(metric)
                    method = self.methods[method_key]
                    mx, my = mat.shape
                    if x_axis is None:
                        x_vec = range(my)
                    y5 = numpy.quantile(mat, q=0.05, axis=0)
                    c5 = [(xi, yi) for (xi, yi) in zip(x_vec, y5)]
                    y95 = numpy.quantile(mat, q=0.95, axis=0)
                    c95 = [(xi, yi) for (xi, yi) in zip(x_vec, y95)]

                    ref1 = f"A{index}"
                    ref2 = f"B{index}"
                    opt1 = f"mark=none, forget plot, opacity=0, name path={ref1}"
                    opt2 = f"mark=none, forget plot, opacity=0, name path={ref2}"

                    ax.append(Plot(coordinates=c5, options=opt1))
                    ax.append(Plot(coordinates=c95, options=opt2))
                    fill_opts = f'{method.color}, forget plot, opacity=0.1'
                    fill_tex = r'\addplot' + f'[{fill_opts}] fill between[of={ref1} and {ref2}];'
                    ax.append(NoEscape(fill_tex))

                    y_med = numpy.median(mat, axis=0)
                    c = [(xi, yi) for (xi, yi) in zip(x_vec, y_med)]
                    plot_options = f"mark=none, color={method.color}, {method.linestyle}"
                    ax.append(Plot(name=method.label, coordinates=c, options=plot_options))

        out_f = join(get_out_dir(), metric).__str__()
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
