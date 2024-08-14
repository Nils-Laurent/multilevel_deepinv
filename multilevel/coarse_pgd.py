import torch
from deepinv.optim.optim_iterators import PGDIteration
from deepinv.optim.optim_iterators.pgd import fStepPGD, gStepPGD
from deepinv.optim.optim_iterators.utils import gradient_descent_step


class CPGDIteration(PGDIteration):
    r"""
    Iterator for Coarse Gradient Descent.
    """

    def __init__(self, coarse_correction=None, **kwargs):
        super().__init__(**kwargs)
        self.coarse_correction = coarse_correction
        self.g_step = gStepPGD(**kwargs)
        self.f_step = fStepPGD(**kwargs)

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        r"""
        Single gradient descent iteration on coarse space.

        :param dict X: Dictionary containing the current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :return: Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev = X["est"][0]
        z = self.f_step(x_prev, cur_data_fidelity, cur_params, y, physics)

        if not(self.coarse_correction is None):
            z -= cur_params["stepsize"] * self.coarse_correction
        assert not torch.isnan(z).any()

        x = self.g_step(z, cur_prior, cur_params)
        return {"est": (x,), "cost": None}
