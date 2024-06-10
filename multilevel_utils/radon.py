import deepinv
import torch


class Tomography(deepinv.physics.LinearPhysics):
    def __init__(self, angles, img_width, circle=False, device=torch.device("cpu"), dtype=torch.float, **kwargs):
        super(Tomography, self).__init__(**kwargs)
        self.angles = angles
        self.img_width = img_width
        self.circle = circle
        self.device = device
        self.dtype = dtype

        self.op = deepinv.physics.Tomography(
            angles=angles,
            img_width=img_width,
            circle=circle,
            device=device,
            dtype=dtype,
        )
        self.norm = self.op.compute_norm(torch.randn(1, 1, img_width, img_width).to(device))
        self.norm = torch.sqrt(self.norm)
        self.radon = self.op.radon

    def A(self, x, **kwargs):
        return self.op.A(x, **kwargs)/self.norm

    def A_adjoint(self, x, **kwargs):
        return self.op.A_adjoint(x, **kwargs)/self.norm

    def A_dagger(self, x, **kwargs):
        return self.op.A_dagger(x*self.norm, **kwargs)