import os

import deepinv.physics.functional
import torch
from deepinv.optim import PnP, Prior, RED
from deepinv.optim.optim_iterators import GDIteration
from deepinv.physics import Inpainting, Blur, BlurFFT, Demosaicing, MRI, Denoising
import deepinv.optim as optim

# multilevel imports
from multilevel.info_transfer import DownsamplingTransfer
from multilevel.coarse_gradient_descent import CGDIteration
import multilevel.iterator as multi_level
from multilevel.prior import TVPrior
from multilevel_utils.radon import Tomography
import torch.nn as nn
import torch.nn.functional as F

import copy

from utils.paths import get_out_dir


class CoarseModel(torch.nn.Module):
    def __init__(self, prior, data_fidelity, fine_physics, ml_params, *args, **kwargs):
        """
        :param multi_level.MultiLevelParams ml_params: all parameters
        """
        super().__init__(*args, **kwargs)
        self.ph = ml_params
        self.pc = self.ph.coarse_params()

        self.g = prior

        if isinstance(self.pc.coarse_prior(), Prior):
            self.g = self.pc.coarse_prior()
        elif isinstance(self.pc.ml_denoiser(), nn.Module):
            raise NotImplementedError("Feature is obsolete")

        self.f = data_fidelity
        self.fph = fine_physics
        self.physics = None
        self.cit_str = self.ph.cit()
        self.cit_op = None

    def projection(self, x):
        if self.cit_op is None:
            self.cit_op = DownsamplingTransfer(x, self.pc.cit(), padding="circular")
            x_proj = self.cit_op.projection(x)

            if self.physics is None:
                self.coarse_physics(x_proj)
            return x_proj

        x_proj = self.cit_op.projection(x)
        return x_proj

    def project_observation(self, y):
        if isinstance(self.fph, Tomography) or isinstance(self.fph, MRI):
            u = self.fph.A_dagger(y)
            v = self.projection(u)
            return self.physics.A(v)

        return self.projection(y)

    def prolongation(self, x):
        if self.cit_op is None:
            self.cit_op = DownsamplingTransfer(x, self.pc.cit(), padding="circular")
        x_prol = self.cit_op.prolongation(x)
        return x_prol

    def grad(self, x, y, physics, params):
        grad_f = self.f.grad(x, y, physics)

        if isinstance(self.pc.coherence_prior(), Prior):
            coherence_prior = self.pc.coherence_prior()
        else:
            coherence_prior = self.g

        if isinstance(coherence_prior, PnP):
            grad_g = x - coherence_prior.denoiser(x, sigma=params.g_param())
        elif hasattr(coherence_prior, 'denoiser'):
            grad_g = coherence_prior.grad(x, sigma_denoiser=params.g_param())
        elif hasattr(coherence_prior, 'moreau_grad'):
            grad_g = coherence_prior.moreau_grad(x, gamma=params.gamma_moreau())
        else:
            raise NotImplementedError("Gradient not defined in this case")

        return grad_f + params.lambda_r() * grad_g

    def coarse_data(self, X, y_h):
        x0_h = X['est']
        if not isinstance(x0_h, torch.Tensor):
            x0_h = x0_h[0]  # primal value of 'est'

        # Projection
        x0 = self.projection(x0_h)
        y = self.project_observation(y_h)

        return x0, x0_h, y

    def coarse_physics(self, x_coarse):
        if isinstance(self.fph, Inpainting) or isinstance(self.fph, Demosaicing):
            m_fine = self.fph.mask.data
            m_coarse = self.projection(m_fine)
            if m_coarse.dim() == 4:
                c_mask = torch.squeeze(m_coarse, 0)
            else:
                c_mask = m_coarse
            self.physics = Inpainting(tensor_size=m_coarse.shape, mask=c_mask, device=m_fine.device)
        elif isinstance(self.fph, Blur) or isinstance(self.fph, BlurFFT):
            half_l = self.pc.cit().get_filter().shape[0] // 2
            f0 = F.pad(self.fph.filter, (half_l,) * 4)
            rep = self.cit_op.filt_2d.repeat(f0.shape[1], 1, 1, 1).to(x_coarse.device)
            filt = F.conv2d(f0, rep, groups=f0.shape[1], padding="valid")
            filt = filt[:, :, :: 2, :: 2]  # downsample
            #filt = F.interpolate(self.fph.filter, scale_factor=0.5, mode='bilinear')
            if isinstance(self.fph, BlurFFT):
                self.physics = BlurFFT(img_size=x_coarse.shape[1:], filter=filt, device=x_coarse.device)
            else:
                self.physics = Blur(filter=filt, padding=self.fph.padding, device=x_coarse.device)
        elif isinstance(self.fph, MRI):
            m_fine = self.fph.mask
            lw = x_coarse.shape[2]
            lh = x_coarse.shape[3]
            m_coarse = m_fine[:, :, lw//2:(lw//2 + lw), lh//2:(lh//2 + lh)]
            #from torchvision.utils import save_image
            #save_image(
            #    m_coarse[0, 0, ::].unsqueeze(0),
            #    os.path.join(get_out_dir(), f"m_coarse{self.pc.level}.png")
            #)
            if m_coarse.dim() == 4:
                c_mask = torch.squeeze(m_coarse, 0)
            else:
                c_mask = m_coarse
            self.physics = MRI(img_size=x_coarse.shape[1:], mask=c_mask, device=m_fine.device)
        elif isinstance(self.fph, Tomography):
            theta_c = self.fph.radon.theta
            size_c = x_coarse.shape[-2]
            self.physics = Tomography(angles=theta_c, img_width=size_c, device=x_coarse.device)
        elif isinstance(self.fph, Denoising):
            self.physics = self.fph
        else:
            raise NotImplementedError("Coarse physics not implemented for " + str(type(self.fph)))

        return self.physics

    def is_large(self, x):
        # check image size
        sz_min = torch.min(torch.tensor(x.shape[2:]))
        cit_len = self.pc.cit().get_filter().shape[0]

        return cit_len < sz_min

    def init_ml_x0(self, X, y_h, grad=None):
        [x0, x0_h, y] = self.coarse_data(X, y_h)
        coarse_iter_class = self.pc.coarse_iterator()

        if self.ph.scale_coherent_gradient_init() is True:
            if grad is None:
                grad_x0 = self.grad(x0_h, y_h, self.fph, self.ph)
            else:
                grad_x0 = grad(x0_h)

            v = self.projection(grad_x0)
            v -= self.grad(x0, y, self.physics, self.pc)

            # Coarse gradient (first order coherent)
            grad_coarse = lambda x: self.grad(x, y, self.physics, self.pc) + v
            level_iteration = coarse_iter_class(coarse_correction=v)
        else:
            grad_coarse = lambda x: self.grad(x, y, self.physics, self.pc)
            level_iteration = coarse_iter_class()

        if self.is_large(x0):
            if self.pc.level > 1:
                model = CoarseModel(self.g, self.f, self.physics, self.pc)
                x0 = model.init_ml_x0({'est': [x0]}, y, grad=grad_coarse)
        else:
            print(f"Warning: Coarse init: image is small, cannot iterate below level {self.pc.level}")

        f_init = lambda def_y, def_ph: {'est': [x0], 'cost': None}
        #iteration = GDIteration(has_cost=False)
        #iteration_class = self.pc.coarse_iterator()
        #iteration = iteration_class()

        model = optim.optim_builder(
            level_iteration,
            data_fidelity=self.f,
            prior=self.g,
            custom_init=f_init,
            max_iter=self.pc.iters_init(),
            params_algo=self.pc.params,
        )
        x_est_coarse = model(y, self.physics)
        return self.prolongation(x_est_coarse)

    def forward(self, X, y_h, grad=None):
        [x0, x0_h, y] = self.coarse_data(X, y_h)
        coarse_iter_class = self.pc.coarse_iterator()
        from torchvision.utils import save_image
        save_image(
            #torch.norm(y[0, ::], dim=0, keepdim=True),
            y[0, ::], os.path.join(get_out_dir(), f"y_coarse{self.pc.level}.png")
        )

        if self.ph.scale_coherent_gradient() is True:
            if grad is None:
                grad_x0 = self.grad(x0_h, y_h, self.fph, self.ph)
            else:
                grad_x0 = grad(x0_h)

            v = self.projection(grad_x0)
            v -= self.grad(x0, y, self.physics, self.pc)

            # Coarse gradient (first order coherent)
            grad_coarse = lambda x: self.grad(x, y, self.physics, self.pc) + v
            level_iteration = coarse_iter_class(coarse_correction=v)
        else:
            grad_coarse = lambda x: self.grad(x, y, self.physics, self.pc)
            level_iteration = coarse_iter_class()

        if self.is_large(x0):
            if self.pc.level > 1:
                model = CoarseModel(self.g, self.f, self.physics, self.pc)
                diff = model({'est': [x0]}, y, grad=grad_coarse)
                x1 = x0 + self.pc.stepsize() * diff
            else:
                x1 = x0
        else:
            print(f"Warning: Coarse model: image is small, cannot iterate below level {self.pc.level}")
            x1 = x0

        if isinstance(self.g, TVPrior):
            self.g.gamma_moreau = self.pc.gamma_moreau()  # compute gradient on Moreau envelope

        f_init = lambda def_y, def_ph: {'est': [x1], 'cost': None}
        model = optim.optim_builder(
            level_iteration,
            data_fidelity=self.f,
            prior=self.g,
            custom_init=f_init,
            max_iter=self.pc.iters(),
            params_algo=self.pc.params,
        )
        x_est_coarse = model(y, self.physics)
        save_image(
            #torch.norm(y[0, ::], dim=0, keepdim=True),
            x_est_coarse[0, ::], os.path.join(get_out_dir(), f"x_coarse{self.pc.level}.png")
        )

        assert not torch.isnan(x_est_coarse).any()

        return self.prolongation(x_est_coarse - x0) # / factor**2

