import deepinv
import torch

from multilevel.info_transfer import DownsamplingTransfer, BlackmannHarris


class ScaleModel(torch.nn.Module):
    def __init__(self, network, pinv=False, ckpt_path=None):
        super().__init__()
        self.network = network
        self.pinv = pinv

        if ckpt_path is not None:
            self.network.load_state_dict(torch.load(ckpt_path), strict=True)
            self.network.eval()

    def forward(self, y, physics, **kwargs):
        if isinstance(physics, torch.nn.DataParallel):
            physics = physics.module

        sigma = physics.noise_model.sigma
        scale = physics.noise_model.scale

        est = self.network(y, sigma, scale)
        for i in range(physics.noise_model.scale_count - 1):
            est = physics.noise_model.prolongation(est)

        return est


class SigmaScaleUniformGaussianNoise(torch.nn.Module):
    def __init__(self, max_scale, sigma_min=0.0, sigma_max=0.5):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cit_filt = BlackmannHarris()
        self.max_scale = max_scale

        if max_scale < 1:
            raise ValueError("max_scale must be greater than or equal to 1")

        # defined later during execution
        self.cit_vec = []         # depends on image size
        self.sigma = None         # has a random value
        self.scale = None         # has a random value
        self.scale_count = None   # has a random value

    def projection(self, x):
        cit = DownsamplingTransfer(x, def_filter=self.cit_filt)
        self.cit_vec.append(cit)
        return cit.projection(x)

    def prolongation(self, x):
        cit = self.cit_vec.pop()
        return cit.prolongation(x)


    def forward(self, x, sigma=None, **kwargs):
        self.sigma = (
            torch.rand((x.shape[0], 1) + (1,) * (x.dim() - 2), device=x.device)
            * (self.sigma_max - self.sigma_min)
            + self.sigma_min
        )

        self.scale_count = 1 * (torch.randint(0, self.max_scale, (1,), device=x.device) + 1)
        self.scale = (
            torch.zeros((x.shape[0], 1) + (1,) * (x.dim() - 2), device=x.device) + 1 / self.scale_count
        )

        noise = torch.randn_like(x) * self.sigma
        xn = x + noise
        for i in range(self.scale_count - 1):
            xn = self.projection(xn)
        return xn


def create_physics(max_scale, noise_pow, device, gpu_num, batch_shape=None):
    physics = deepinv.physics.DecomposablePhysics()
    physics.noise_model = SigmaScaleUniformGaussianNoise(max_scale=max_scale, sigma_max=noise_pow)

    if gpu_num > 1:

        x_init_physics = torch.randn(batch_shape).to(device)

        physics = deepinv.physics.DecomposablePhysics()
        physics.noise_model = ParallelGaussianNoise(sigma_max=noise_pow, x_shape=x_init_physics.shape,
                                                    x_device=x_init_physics.device)
        physics = physics.to(device)
        physics = torch.nn.DataParallel(physics, device_ids=list(range(gpu_num)))

    return physics

