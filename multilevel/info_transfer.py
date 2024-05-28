import torch
import deepinv


class InfoTransfer:
    def __init__(self, x):
        pass

    def build_cit_matrices(self, x):
        raise NotImplementedError("cit_matrices not overridden")

    def get_cit_matrices(self):
        if self.cit_c is None or self.cit_r is None:
            raise ValueError("CIT matrix is none")
        return self.cit_c, self.cit_r

    def projection(self, x):
        raise NotImplementedError("projection not overridden")

    def prolongation(self, x):
        raise NotImplementedError("prolongation not overridden")


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


class BlackmannHarris:
    def __str__(self):
        return 'blackmannharris'

    def get_filter(self):
        # 8 coefficients
        k0 = torch.tensor(
            [0.0001, 0.0334, 0.3328, 0.8894,
             0.8894, 0.3328, 0.0334, 0.0001]
        )
        return k0


class DownsamplingTransfer(InfoTransfer):
    def __init__(self, x, def_filter=BlackmannHarris()):
        super().__init__(x)

        match def_filter:
            case Kaiser():
                pass
            case BlackmannHarris():
                pass
            case _:
                raise NotImplementedError("Downsampling filter not implemented")

        k0 = def_filter.get_filter()
        filt_2d = self.set_2d_filter(k0, x.dtype)
        if len(x.shape) == 3:
            shape = x.shape
        else:
            shape = x.shape[1:]
        self.op = deepinv.physics.Downsampling(shape, filter=filt_2d, factor=2, device=x.device, padding="circular")

    def set_2d_filter(self, k0, dtype):
        k0 = k0 / torch.sum(k0)
        k_filter = torch.outer(k0, k0)
        k_filter = k_filter.unsqueeze(0).unsqueeze(0).type(dtype)
        return k_filter

    def projection(self, x):
        return self.op.A(x)

    def prolongation(self, x):
        return self.op.A_adjoint(x)
