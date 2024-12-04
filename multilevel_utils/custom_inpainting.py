import torch
from deepinv.physics import Inpainting

class CInpainting(Inpainting):
    def forward(self, x, **kwargs):
        t1 = torch.ones_like(x)


        # replace black inpainting pixels by gray ones
        in_half = self.A(x) + 0.5 * (t1 - self.A(t1))
        return in_half
        return self.noise_model(in_half)