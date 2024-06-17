from os.path import join

import numpy
from matplotlib import pyplot

from utils.paths import get_out_dir


class GenFigMetricLogger:
    def __init__(self):
        self.log_vec = []
        self.label_vec = []

    def add_logger(self, x, label):
        self.log_vec.append(x)
        self.label_vec.append(label)

    def gen_fig(self, metric):

        fig, ax = pyplot.subplots()

        for g, l in zip(self.log_vec, self.label_vec):
            #if not (metric in g.keys()):
            #    continue

            m = g.metric_matrix(metric)
            mx, my = m.shape
            y5 = numpy.quantile(m, q=0.05, axis=0)
            y95 = numpy.quantile(m, q=0.95, axis=0)
            ax.fill_between(range(my), y5, y95, alpha=0.1, color='goldenrod')
            #ax.fill_between(sigma_vec, y25, y75, alpha=0.1, color='darkorchid')
        pyplot.xlabel("x")
        pyplot.ylabel("y")
        pyplot.show()

        out_path = get_out_dir()
        pyplot.savefig(join(out_path, (metric + ".png")))
        pyplot.close('all')
