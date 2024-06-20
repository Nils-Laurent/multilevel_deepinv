from os.path import join

import numpy
from matplotlib import pyplot

from utils.paths import get_out_dir

from dataclasses import dataclass

@dataclass
class MRed:
    key = 'RED'
    color = 'red'
    linestyle = '--'
    label = key
@dataclass
class MRedML:
    key = 'RED_ML'
    color = 'red'
    linestyle = '-'
    label = key
@dataclass
class MRedInit:
    key = 'RED_INIT'
    color = 'black'
    linestyle = '--'
    label = key
@dataclass
class MRedMLInit:
    key = 'RED_ML_INIT'
    color = 'black'
    linestyle = '-'
    label = key
@dataclass
class MDPIR:
    key = 'DPIR'
    color = 'green'
    linestyle = '-'
    label = key
@dataclass
class MFb:
    key = 'FB_TV'
    color = 'blue'
    linestyle = '--'
    label = key
@dataclass
class MFbML:
    key = 'FB_TV_ML'
    color = 'blue'
    linestyle = '-'
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

    def gen_fig(self, metric):
        fig, ax = pyplot.subplots()

        for mkey, g in self.data.items():
            if not g.has_key(metric) or not self.is_valid_key(mkey):
                continue

            mat = g.metric_matrix(metric)
            method = self.methods[mkey]
            mx, my = mat.shape
            y5 = numpy.quantile(mat, q=0.05, axis=0)
            y95 = numpy.quantile(mat, q=0.95, axis=0)
            y_med = numpy.median(mat, axis=0)
            ax.fill_between(range(my), y5, y95, alpha=0.15, color=method.color)
            pyplot.plot(range(my), y_med, color=method.color, linestyle=method.linestyle, label=method.label)
            #ax.fill_between(range(my), y5, y95, alpha=0.1, color='goldenrod')
            #ax.fill_between(sigma_vec, y25, y75, alpha=0.1, color='darkorchid')
        pyplot.xlabel("x")
        pyplot.ylabel("y")
        pyplot.show()

        out_path = get_out_dir()
        pyplot.savefig(join(out_path, (metric + ".png")))
        pyplot.close('all')
