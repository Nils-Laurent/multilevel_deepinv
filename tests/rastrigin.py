import numpy as np
from matplotlib import pylab


def rastrigin(x):
    n = len(x)  # 2D
    A = 10

    y = A * n
    for i in range(0, n):
        y += x[i] ** 2 - A * np.cos(2 * np.pi * x[i])

    return y

def eval_rastrigin(sup_range, n_sample):
    x0_space = np.linspace(-sup_range, sup_range, n_sample)
    x1_space = np.linspace(-sup_range, sup_range, n_sample)

    x0_mg, x1_mg = np.meshgrid(x0_space, x1_space)
    x_mg = np.array([x0_mg, x1_mg])
    y_mg = rastrigin(x_mg)

    im = pylab.imshow(y_mg, cmap=pylab.cm.coolwarm, vmin=0, vmax=80)  # drawing the function
    i_range = range(0, n_sample, int(n_sample/8))
    pylab.xticks(i_range, np.around(x0_space[i_range], decimals=1))
    pylab.yticks(i_range, np.around(x1_space[i_range], decimals=1))
    # adding the Contour lines with labels
    cset = pylab.contour(y_mg, [15, 30, 45, 60, 75], linewidths=1, cmap=pylab.cm.summer)
    pylab.clabel(cset, inline=True, fmt='%1.1f', fontsize=6)
    pylab.colorbar(im)  # adding the colobar on the right
    # latex fashion title
    pylab.title(f"{n_sample} samples: $z=nA + \sum x_i^2 - A cos(2 \pi x_i)$")
    pylab.show()


def test_rastrigin():
    sample0 = 256
    for i in range(0, 5):
        sample = int(sample0 / (2 ** i))
        eval_rastrigin(5.12, sample)
