import torch
import torch.nn as nn
from deepinv.optim.utils import gradient_descent
from deepinv.optim import DataFidelity
from deepinv.physics import PoissonNoise


def to_nn_parameter(x):
    if isinstance(x, torch.Tensor):
        return torch.nn.Parameter(x, requires_grad=False)
    else:
        return torch.nn.Parameter(torch.tensor(x), requires_grad=False)

class CPoissonNoise(PoissonNoise):
    r"""

    Poisson noise :math:`y = \mathcal{P}(\frac{x}{\gamma})`
    with gain :math:`\gamma>0`.

    If ``normalize=True``, the output is multiplied by the gain, i.e., :math:`\tilde{y} = \gamma y`.

    |sep|

    :Examples:

        Adding Poisson noise to a physics operator by setting the ``noise_model``
        attribute of the physics operator:

        >>> from deepinv.physics import Denoising, PoissonNoise
        >>> import torch
        >>> physics = Denoising()
        >>> physics.noise_model = PoissonNoise()
        >>> x = torch.rand(1, 1, 2, 2)
        >>> y = physics(x)

    :param float gain: gain of the noise.
    :param bool normalize: normalize the output.
    :param bool clip_positive: clip the input to be positive before adding noise. This may be needed when a NN outputs negative values e.g. when using LeakyReLU.
    :param torch.Generator (Optional) rng: a pseudorandom random number generator for the parameter generation.

    """

    def __init__(
        self, gain=1.0, normalize=True, clip_positive=False, rng: torch.Generator = None
    ):
        super().__init__(rng=rng)
        self.normalize = to_nn_parameter(normalize)
        self.update_parameters(gain=gain)
        self.clip_positive = clip_positive

    def forward(self, x, gain=None, seed: int = None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param None, float, torch.Tensor gain: gain of the noise. If not None, it will overwrite the current noise level.
        :param int seed: the seed for the random number generator, if `rng` is provided.

        :returns: noisy measurements
        """
        self.update_parameters(gain=gain)
        self.rng_manual_seed(seed)
        y = torch.poisson(
            torch.clip(x / self.gain, min=0.0) if self.clip_positive else x / self.gain,
            generator=self.rng,
        )
        if self.normalize:
            y *= self.gain
        return y

    def update_parameters(self, gain=None, **kwargs):
        r"""
        Updates the gain of the noise.

        :param float, torch.Tensor gain: gain of the noise.
        """
        if gain is not None:
            self.gain = to_nn_parameter(gain)

class Potential(nn.Module):
    r"""
    Base class for a potential :math:`h : \xset \to \mathbb{R}` to be used in an optimization problem.

    Comes with methods to compute the potential gradient, its proximity operator, its convex conjugate (and associated gradient and prox).

    :param callable fn: Potential function :math:`h(x)` to be used in the optimization problem.
    """

    def __init__(self, fn=None):
        super().__init__()
        self._fn = fn

    def fn(self, x, *args, **kwargs):
        r"""
        Computes the value of the potential :math:`h(x)`.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) prior :math:`h(x)`.
        """
        return self._fn(x, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        r"""
        Computes the value of the potential :math:`h(x)`.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) prior :math:`h(x)`.
        """
        return self.fn(x, *args, **kwargs)

    def conjugate(self, x, *args, **kwargs):
        r"""
        Computes the convex conjugate potential :math:`h^*(y) = \sup_{x} \langle x, y \rangle - h(x)`.
        By default, the conjugate is computed using internal gradient descent.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`h^*(y)`.
        """
        grad = lambda z: self.grad(z, *args, **kwargs) - x
        z = gradient_descent(-grad, x)
        return self.forward(z, *args, **kwargs) - torch.sum(
            x.reshape(x.shape[0], -1) * z.reshape(z.shape[0], -1), dim=-1
        ).view(x.shape[0], 1)

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the potential term :math:`h` at :math:`x`.
        By default, the gradient is computed using automatic differentiation.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h`, computed in :math:`x`.
        """
        with torch.enable_grad():
            x = x.requires_grad_()
            h = self.forward(x, *args, **kwargs)
            grad = torch.autograd.grad(
                h, x, torch.ones_like(h), create_graph=True, only_inputs=True
            )[0]
        return grad

    def grad_conj(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the convex conjugate potential :math:`h^*` at :math:`x`.
        If the potential is convex and differentiable, the gradient of the conjugate is the inverse of the gradient of the potential.
        By default, the gradient is computed using automatic differentiation.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h^*`, computed in :math:`x`.
        """
        with torch.enable_grad():
            x = x.requires_grad_()
            h = self.conjugate(x, *args, **kwargs)
            grad = torch.autograd.grad(
                h,
                x,
                torch.ones_like(h),
                create_graph=True,
                only_inputs=True,
            )[0]
        return grad

    def prox(
        self,
        x,
        *args,
        gamma=1.0,
        stepsize_inter=1.0,
        max_iter_inter=50,
        tol_inter=1e-3,
        **kwargs,
    ):
        r"""
        Calculates the proximity operator of :math:`h` at :math:`x`. By default, the proximity operator is computed using internal gradient descent.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param float stepsize_inter: stepsize used for internal gradient descent
        :param int max_iter_inter: maximal number of iterations for internal gradient descent.
        :param float tol_inter: internal gradient descent has converged when the L2 distance between two consecutive iterates is smaller than tol_inter.
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma h}(x)`, computed in :math:`x`.
        """
        grad = lambda z: gamma * self.grad(z, *args, **kwargs) + (z - x)
        return gradient_descent(
            grad, x, step_size=stepsize_inter, max_iter=max_iter_inter, tol=tol_inter
        )

    def prox_conjugate(self, x, *args, gamma=1.0, lamb=1.0, **kwargs):
        r"""
        Calculates the proximity operator of the convex conjugate :math:`(\lambda h)^*` at :math:`x`, using the Moreau formula.

        ::Warning:: Only valid for convex potential.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param float lamb: math:`\lambda` parameter in front of :math:`f`
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma \lambda h)^*}(x)`, computed in :math:`x`.
        """
        return x - gamma * self.prox(x / gamma, *args, gamma=lamb / gamma, **kwargs)

    def bregman_prox(
        self,
        x,
        bregman_potential,
        *args,
        gamma=1.0,
        stepsize_inter=1.0,
        max_iter_inter=50,
        tol_inter=1e-3,
        **kwargs,
    ):
        r"""
        Calculates the (right) Bregman proximity operator of h` at :math:`x`, with Bregman potential `bregman_potential`.

        .. math::

            \operatorname{prox}^h_{\gamma \regname}(x) = \underset{u}{\text{argmin}} \frac{\gamma}{2}h(u) + D_\phi(u,x)

        where :math:`D_\phi(x,y)` stands for the Bregman divergence with potential :math:`\phi`.

        By default, the proximity operator is computed using internal gradient descent.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param dinv.optim.bregman.Bregman bregman_potential: Bregman potential to be used in the Bregman proximity operator.
        :param float gamma: stepsize of the proximity operator.
        :param float stepsize_inter: stepsize used for internal gradient descent
        :param int max_iter_inter: maximal number of iterations for internal gradient descent.
        :param float tol_inter: internal gradient descent has converged when the L2 distance between two consecutive iterates is smaller than tol_inter.
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}^h_{\gamma \regname}(x)`, computed in :math:`x`.
        """
        grad = lambda u: gamma * self.grad(u, *args, **kwargs) + (
            bregman_potential.grad(u) - bregman_potential.grad(x)
        )
        return gradient_descent(
            grad, x, step_size=stepsize_inter, max_iter=max_iter_inter, tol=tol_inter
        )

class Distance(Potential):
    r"""
    Distance :math:`\distance{x}{y}`.

    This is the base class for a distance :math:`\distance{x}{y}` between a variable :math:`x` and an observation :math:`y`.
    Comes with methods to compute the distance gradient, proximal operator or convex conjugate with respect to the variable :math:`x`.

    .. warning::
        All variables have a batch dimension as first dimension.

    :param callable d: distance function :math:`\distance{x}{y}`. Outputs a tensor of size `B`, the size of the batch. Default: None.
    """

    def __init__(self, d=None):
        super().__init__(fn=d)

    def fn(self, x, y, *args, **kwargs):
        r"""
        Computes the distance :math:`\distance{x}{y}`.

        :param torch.Tensor x: Variable :math:`x`.
        :param torch.Tensor y: Observation :math:`y`.
        :return: (torch.Tensor) distance :math:`\distance{x}{y}` of size `B` with `B` the size of the batch.
        """
        return self._fn(x, y, *args, **kwargs)

    def forward(self, x, y, *args, **kwargs):
        r"""
        Computes the value of the distance :math:`\distance{x}{y}`.

        :param torch.Tensor x: Variable :math:`x`.
        :param torch.Tensor y: Observation :math:`y`.
        :return: (torch.Tensor) distance :math:`\distance{x}{y}` of size `B` with `B` the size of the batch.
        """
        return self.fn(x, y, *args, **kwargs)


class PoissonLikelihoodDistance(Distance):
    r"""
    (Negative) Log-likelihood of the Poisson distribution.

    .. math::

        \d{y}{x} =  \sum_i y_i \log(y_i / x_i) + x_i - y_i


    .. note::

        The function is not Lipschitz smooth w.r.t. :math:`x` in the absence of background (:math:`\beta=0`).

    :param float gain: gain of the measurement :math:`y`. Default: 1.0.
    :param float bkg: background level :math:`\beta`. Default: 0.
    :param bool denormalize: if True, the measurement is divided by the gain. By default, in the class :class:`physics.noise.PoissonNoise`, the measurements are multiplied by the gain after being sampled by the Poisson distribution. Default: True.
    """

    def __init__(self, gain=1.0, bkg=0, denormalize=False):
        super().__init__()
        self.bkg = bkg
        self.gain = gain
        self.denormalize = denormalize

    def fn(self, x, y, *args, **kwargs):
        r"""
        Computes the Kullback-Leibler divergence

        :param torch.Tensor x: Variable :math:`x` at which the distance is computed.
        :param torch.Tensor y: Observation :math:`y`.
        """
        if self.denormalize:
            y = y / self.gain
        return (-y * torch.log(x / self.gain + self.bkg)).flatten().sum() + (
            (x / self.gain) + self.bkg - y
        ).reshape(x.shape[0], -1).sum(dim=1)

    def grad(self, x, y, *args, **kwargs):
        r"""
        Gradient of the Kullback-Leibler divergence

        :param torch.Tensor x: signal :math:`x` at which the function is computed.
        :param torch.Tensor y: measurement :math:`y`.
        """
        if self.denormalize:
            y = y / self.gain
        return self.gain * (torch.ones_like(x) - y / (x / self.gain + self.bkg))

    def prox(self, x, y, *args, gamma=1.0, **kwargs):
        r"""
        Proximal operator of the Kullback-Leibler divergence

        :param torch.Tensor x: signal :math:`x` at which the function is computed.
        :param torch.Tensor y: measurement :math:`y`.
        :param float gamma: proximity operator step size.
        """
        if self.denormalize:
            y = y / self.gain
        out = (
            x
            - (1 / (self.gain * gamma))
            * ((x - (1 / (self.gain * gamma))).pow(2) + 4 * y / gamma).sqrt()
        )
        return out / 2


class LogPoissonLikelihoodDistance(Distance):
    r"""
    Log-Poisson negative log-likelihood.

    .. math::

        \distancz{z}{y} =  N_0 (1^{\top} \exp(-\mu z)+ \mu \exp(-\mu y)^{\top}x)

    Corresponds to LogPoissonNoise with the same arguments N0 and mu.
    There is no closed-form of the prox known.

    :param float N0: average number of photons
    :param float mu: normalization constant
    """

    def __init__(self, N0=1024.0, mu=1 / 50.0):
        super().__init__()
        self.mu = mu
        self.N0 = N0

    def fn(self, x, y, *args, **kwargs):
        out1 = torch.exp(-x * self.mu) * self.N0
        out2 = torch.exp(-y * self.mu) * self.N0 * (x * self.mu)
        return (out1 + out2).reshape(x.shape[0], -1).sum(dim=1)


class CPoissonLikelihood(DataFidelity):
    r"""

    Poisson negative log-likelihood.

    .. math::

        \datafid{z}{y} =  -y^{\top} \log(z+\beta)+1^{\top}z

    where :math:`y` are the measurements, :math:`z` is the estimated (positive) density and :math:`\beta\geq 0` is
    an optional background level.

    .. note::

        The function is not Lipschitz smooth w.r.t. :math:`z` in the absence of background (:math:`\beta=0`).

    :param float gain: gain of the measurement :math:`y`. Default: 1.0.
    :param float bkg: background level :math:`\beta`. Default: 0.
    :param bool denormalize: if True, the measurement is multiplied by the gain. Default: True.
    """

    def __init__(self, gain=1.0, bkg=0, denormalize=True):
        d = PoissonLikelihoodDistance(gain=gain, bkg=bkg, denormalize=denormalize)
        super().__init__(d=d)
        self.d = d
        self.bkg = bkg
        self.gain = gain
        self.normalize = denormalize
