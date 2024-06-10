import torch
from deepinv.unfolded import BaseUnfold


class BaseUnfoldGradSplit(BaseUnfold):
    def __init__(self, *args, total_iters, grad_iters, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_iters = total_iters
        self.grad_iters = grad_iters

    def forward(self, y, physics, x_gt=None, compute_metrics=False):
        r"""
        Runs the fixed-point iteration algorithm for solving :ref:`(1) <optim>`.

        :param torch.Tensor y: measurement vector.
        :param deepinv.physics physics: physics of the problem for the acquisition of ``y``.
        :param torch.Tensor x_gt: (optional) ground truth image, for plotting the PSNR across optim iterations.
        :param bool compute_metrics: whether to compute the metrics or not. Default: ``False``.
        :return: If ``compute_metrics`` is ``False``,  returns (torch.Tensor) the output of the algorithm.
                Else, returns (torch.Tensor, dict) the output of the algorithm and the metrics.
        """
        with torch.no_grad():
            X, metrics = self.fixed_point(
                y, physics, x_gt=x_gt, compute_metrics=compute_metrics
            )
        x = self.get_output(X)
        if compute_metrics:
            return x, metrics
        else:
            return x

def unfolded_builder(
    iteration,
    params_algo={"lambda": 1.0, "stepsize": 1.0},
    data_fidelity=None,
    prior=None,
    F_fn=None,
    g_first=False,
    **kwargs,
):
    r"""
    Helper function for building an instance of the :meth:`BaseUnfold` class.

    :param str, deepinv.optim.OptimIterator iteration: either the name of the algorithm to be used,
        or directly an optim iterator.
        If an algorithm name (string), should be either ``"PGD"`` (proximal gradient descent), ``"ADMM"`` (ADMM),
        ``"HQS"`` (half-quadratic splitting), ``"CP"`` (Chambolle-Pock) or ``"DRS"`` (Douglas Rachford).
    :param dict params_algo: dictionary containing all the relevant parameters for running the algorithm,
                            e.g. the stepsize, regularisation parameter, denoising standard deviation.
                            Each value of the dictionary can be either Iterable (distinct value for each iteration) or
                            a single float (same value for each iteration).
                            Default: ``{"stepsize": 1.0, "lambda": 1.0}``. See :any:`optim-params` for more details.
    :param list, deepinv.optim.DataFidelity: data-fidelity term.
                            Either a single instance (same data-fidelity for each iteration) or a list of instances of
                            :meth:`deepinv.optim.DataFidelity` (distinct data-fidelity for each iteration). Default: `None`.
    :param list, deepinv.optim.Prior prior: regularization prior.
                            Either a single instance (same prior for each iteration) or a list of instances of
                            deepinv.optim.Prior (distinct prior for each iteration). Default: `None`.
    :param callable F_fn: Custom user input cost function. default: None.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: False
    :param kwargs: additional arguments to be passed to the :meth:`BaseUnfold` class.
    """
    iterator = create_iterator(iteration, prior=prior, F_fn=F_fn, g_first=g_first)
    return BaseUnfold(
        iterator,
        has_cost=iterator.has_cost,
        data_fidelity=data_fidelity,
        prior=prior,
        params_algo=params_algo,
        **kwargs,
    )