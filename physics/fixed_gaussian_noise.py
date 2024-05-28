import torch
from deepinv.physics.noise import to_nn_parameter


class FixedGaussianNoise(torch.nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = to_nn_parameter(sigma)

    def forward(self, x, sigma=None, **kwargs):
        if sigma is not None:
            sigma = to_nn_parameter(sigma)
            acc = sigma
            for i in range(x.get_device()):
                acc = torch.cat((acc, sigma), dim=0)
            self.sigma = to_nn_parameter(acc)

        return x + torch.randn_like(x) * sigma[(...,) + (None,) * (x.ndim - 1)]

    def forward3(self, x, **kwargs):
        r"""
        Adds the noise to measurements x.

        :param torch.Tensor x: measurements.
        :returns: noisy measurements.
        """

        sigma = (
            torch.rand((x.shape[0], 1) + (1,) * (x.dim() - 2), device=x.device)
            * (self.sigma_max - self.sigma_min)
            + self.sigma_min
        )
        noise = torch.randn_like(x) * sigma
        return x + noise