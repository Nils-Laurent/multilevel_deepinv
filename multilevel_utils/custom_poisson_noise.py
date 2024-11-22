import torch
from deepinv.optim import DataFidelity
from deepinv.physics.noise import PoissonNoise

def to_nn_parameter(x):
    if isinstance(x, torch.Tensor):
        return torch.nn.Parameter(x, requires_grad=False)
    else:
        return torch.nn.Parameter(torch.tensor(x), requires_grad=False)

class CPoissonNoise(PoissonNoise):
    r"""

    Poisson noise :math:`y = \mathcal{P}(\frac{x}{\gamma})`
    with gain :math:`\gamma>0`.

    If ``normalize=True``, the output is divided by the gain, i.e., :math:`\tilde{y} = \gamma y`.

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
        self, gain=1.0, bkg=0.0, normalize=False, clip_positive=False, rng: torch.Generator = None
    ):
        super().__init__(rng=rng)
        self.normalize = to_nn_parameter(normalize)
        self.update_parameters(gain=gain, bkg=bkg)
        self.clip_positive = clip_positive

    def forward(self, x, gain=None, seed: int = None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param None, float, torch.Tensor gain: gain of the noise. If not None, it will overwrite the current noise level.
        :param int seed: the seed for the random number generator, if `rng` is provided.

        :returns: noisy measurements
        """
        self.update_parameters(gain)
        self.rng_manual_seed(seed)
        y = torch.poisson(x / self.gain + self.bkg, generator=self.rng)
        if self.clip_positive or self.normalize:
            assert False  # unsuported parameters
        #y = torch.poisson(
        #    torch.clip(x / self.gain + self.bkg, min=0.0) if self.clip_positive else (x / self.gain + self.bkg),
        #    generator=self.rng,
        #)
        #if self.normalize:
        #    y *= self.gain
        return y

    def update_parameters(self, gain=None, bkg=None, **kwargs):
        r"""
        Updates the gain of the noise.

        :param float, torch.Tensor gain: gain of the noise.
        """
        if gain is not None:
            self.gain = to_nn_parameter(gain)
        if bkg is not None:
            self.bkg = to_nn_parameter(bkg)





class CPoissonLikelihood(DataFidelity):
    r"""

    Poisson negative log-likelihood.

    .. math::

        \datafid{z}{y} =  -y^{\top} \log(z+\beta)+1^{\top}z

    where :math:`y` are the measurements, :math:`z` is the estimated (positive) density and :math:`\beta\geq 0` is
    an optional background level.

    .. note::

        The function is not Lipschitz smooth w.r.t. :math:`z` in the absence of background (:math:`\beta=0`).

    :param float bkg: background level :math:`\beta`.
    """

    def __init__(self, gain=1.0, bkg=0, normalize=False):
        super().__init__()
        self.bkg = bkg
        self.gain = gain
        self.normalize = normalize

    def d(self, x, y):
        r"""
        Computes the Poisson negative log-likelihood.

        :param torch.Tensor x: signal :math:`x` at which the function is computed.
        :param torch.Tensor y: measurement :math:`y`.
        """
        if self.normalize:
            assert False
        #if self.normalize:
        #    y = y * self.gain
        return (-y * torch.log(x / self.gain + self.bkg)).flatten().sum() + (
            x / self.gain + self.bkg
        ).reshape(x.shape[0], -1).sum(dim=1)

    def grad_d(self, x, y):
        r"""
        Gradient of the Poisson negative log-likelihood.


        :param torch.Tensor x: signal :math:`x` at which the function is computed.
        :param torch.Tensor y: measurement :math:`y`.
        """
        if self.normalize:
            assert False
        #if self.normalize:
        #    y = y * self.gain
        return - y / (x + self.gain * self.bkg) + torch.ones_like(x) / self.gain

    def prox_d(self, x, y, gamma=1.0):
        r"""
        Proximal operator of the Poisson negative log-likelihood.

        :param torch.Tensor x: signal :math:`x` at which the function is computed.
        :param torch.Tensor y: measurement :math:`y`.
        :param float gamma: proximity operator step size.
        """
        assert False
