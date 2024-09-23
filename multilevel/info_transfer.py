import numpy
import torch
import deepinv


class DownsamplingTransfer:
    def __init__(self, x, def_filter, padding="circular"):
        if isinstance(def_filter, SincFilter):
            self.filt_2d = def_filter.get_filter_2d()
            self.filt_2d.to(x.device)
        else:
            k0 = def_filter.get_filter()
            self.filt_2d = self.set_2d_filter(k0, x.dtype)

        if len(x.shape) == 3:
            shape = x.shape
        else:
            shape = x.shape[1:]

        if isinstance(def_filter, Dirac):
            padding = "valid"

        self.factor = 2
        self.op = deepinv.physics.Downsampling(shape, filter=self.filt_2d, factor=self.factor, device=x.device, padding=padding)

    def set_2d_filter(self, k0, dtype):
        #k0 = k0 / torch.sum(k0)
        k_filter = torch.outer(k0, k0)
        k_filter = k_filter.unsqueeze(0).unsqueeze(0).type(dtype)
        return k_filter

    def projection(self, x, padding="circular"):
        if x.dim() == 3:  # useful for projecting masks
            x2 = x.unsqueeze(0)
            return self.op.A(x2).squeeze(0)
        return self.op.A(x)

    def prolongation(self, x):
        return self.op.A_adjoint(x)


# ==========================
#       filter list
# ==========================


class Kaiser:
    def __str__(self):
        return 'kaiser'

    def get_filter(self):
        # N = 10
        # beta = 10.0
        k0 = torch.tensor([
            0.0004, 0.0310, 0.2039, 0.5818, 0.9430,
            0.9430, 0.5818, 0.2039, 0.0310, 0.0004
        ])
        return k0

class SincFilter:
    def __str__(self):
        return 'sinc'

    def get_filter_2d(self):
        return deepinv.physics.blur.sinc_filter(factor=2, length=11, windowed=True)

    def get_filter(self):
        f = self.get_filter_2d()
        return f[0, 0, 5, :]

class CFir:  # custom FIR filter
    def __str__(self):
        return 'cfir'

    def get_filter(self):
        # order + 1 coefficients
        k0 = torch.tensor([
            -0.015938026, 0.000019591, 0.013033937, -0.000004666, -0.018657837, 0.000020187, 0.026570831, 0.000002218,
            -0.038348155, 0.000018390, 0.058441238, 0.000007421, -0.102893218, 0.000011707, 0.317258819, 0.500004593,
            0.317258819, 0.000011707, -0.102893218, 0.000007421, 0.058441238, 0.000018390, -0.038348155, 0.000002218,
            0.026570831, 0.000020187, -0.018657837, -0.000004666, 0.013033937, 0.000019591, -0.015938026
        ])
        #k0 = torch.tensor([
        #    -0.000068106, 0.111025515, 0.000061827, -0.103275087, 0.000049373, 0.317230919, 0.499913812, 0.317230919,
        #    0.000049373, -0.103275087, 0.000061827, 0.111025515, -0.000068106
        #])
        return k0


class BlackmannHarris:
    def __str__(self):
        return 'blackmannharris'

    def get_filter(self):
        # 8 coefficients
        #k0 = torch.tensor(
        #    [0.0001, 0.0334, 0.3328, 0.8894,
        #     0.8894, 0.3328, 0.0334, 0.0001]
        #)
        k0 = torch.tensor([
            3.9818e-05, 1.3299e-02, 1.3252e-01, 3.5415e-01, 3.5415e-01, 1.3252e-01, 1.3299e-02, 3.9818e-05]
        )
        return k0

class Dirac:
    def __str__(self):
        return 'dirac'

    def get_filter(self):
        k0 = torch.tensor([1.0])
        return k0
