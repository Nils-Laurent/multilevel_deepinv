import torch
from deepinv.optim.optim_iterators import GDIteration
from deepinv.optim.optim_iterators.utils import gradient_descent_step


class CGDIteration(GDIteration):
    r"""
    Iterator for Coarse Gradient Descent.
    """

    def __init__(self, coherent_grad_fn=None, **kwargs):
        super(GDIteration, self).__init__(**kwargs)
        self.coherent_grad_fn = coherent_grad_fn

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
        grad = cur_params["stepsize"] * self.coherent_grad_fn(x_prev)
        x = gradient_descent_step(x_prev, grad)
        F = self.F_fn(x, cur_prior, cur_params, y, physics) if self.has_cost else None
        return {"est": (x,), "cost": F}
