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
from multilevel.prior import TVPrior as CustTV
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
        self.par_f = ml_params
        self.params = self.par_f.coarse_params()

        self.gfine = prior
        self.g = prior

        if isinstance(self.params.coarse_prior(), Prior):
            self.g = self.params.coarse_prior()
        elif isinstance(self.params.ml_denoiser(), nn.Module):
            raise NotImplementedError("Feature is obsolete")

        self.f = data_fidelity
        self.ph_f = fine_physics
        self.physics = None
        self.cit_str = self.par_f.cit()
        self.cit_op = None

    def projection(self, x):
        if self.cit_op is None:
            self.cit_op = DownsamplingTransfer(x, self.params.cit(), padding="circular")
            x_proj = self.cit_op.projection(x)

            if self.physics is None:
                self.coarse_physics(x_proj)
            return x_proj

        x_proj = self.cit_op.projection(x)
        return x_proj

    def project_observation(self, y):
        if isinstance(self.ph_f, Tomography) or isinstance(self.ph_f, MRI):
            u = self.ph_f.A_dagger(y)
            v = self.projection(u)
            return self.physics.A(v)

        return self.projection(y)

    def prolongation(self, x):
        if self.cit_op is None:
            self.cit_op = DownsamplingTransfer(x, self.params.cit(), padding="circular")
        x_prol = self.cit_op.prolongation(x)
        return x_prol

    def grad(self, x, y, physics, grad_prior, params):
        grad_f = self.f.grad(x, y, physics)

        if isinstance(self.params.coherence_prior(), Prior):
            assert False # should not be used

        # todo : A VALIDER
        if isinstance(grad_prior, PnP):
            #print(f"Grad: : Id - denoiser")
            grad_g = x - grad_prior.denoiser(x, sigma=params.g_param())
        elif hasattr(grad_prior, 'denoiser'):
            grad_g = grad_prior.grad(x, sigma_denoiser=params.g_param())
        elif hasattr(grad_prior, 'moreau_grad'):
            grad_g = grad_prior.moreau_grad(x, gamma=params.gamma_moreau())
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
        if isinstance(self.ph_f, Inpainting) or isinstance(self.ph_f, Demosaicing):
            m_fine = self.ph_f.mask.data
            m_coarse = self.projection(m_fine)
            if m_coarse.dim() == 4:
                c_mask = torch.squeeze(m_coarse, 0)
            else:
                c_mask = m_coarse
            self.physics = Inpainting(tensor_size=m_coarse.shape, mask=c_mask, device=m_fine.device)
        elif isinstance(self.ph_f, Blur) or isinstance(self.ph_f, BlurFFT):
            half_l = self.params.cit().get_filter().shape[0] // 2
            f0 = F.pad(self.ph_f.filter, (half_l,) * 4)
            rep = self.cit_op.filt_2d.repeat(f0.shape[1], 1, 1, 1).to(x_coarse.device)
            filt = F.conv2d(f0, rep, groups=f0.shape[1], padding="valid")
            filt = filt[:, :, :: 2, :: 2]  # downsample
            #filt = F.interpolate(self.fph.filter, scale_factor=0.5, mode='bilinear')
            if isinstance(self.ph_f, BlurFFT):
                self.physics = BlurFFT(img_size=x_coarse.shape[1:], filter=filt, device=x_coarse.device)
            else:
                self.physics = Blur(filter=filt, padding=self.ph_f.padding, device=x_coarse.device)
        elif isinstance(self.ph_f, MRI):
            m_fine = self.ph_f.mask
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
        elif isinstance(self.ph_f, Tomography):
            theta_c = self.ph_f.radon.theta
            size_c = x_coarse.shape[-2]
            self.physics = Tomography(angles=theta_c, img_width=size_c, device=x_coarse.device)
        elif isinstance(self.ph_f, Denoising):
            self.physics = self.ph_f
        else:
            raise NotImplementedError("Coarse physics not implemented for " + str(type(self.ph_f)))

        return self.physics

    def is_large(self, x):
        # check image size
        sz_min = torch.min(torch.tensor(x.shape[2:]))
        cit_len = self.params.cit().get_filter().shape[0]

        return cit_len < sz_min

    def init_ml_x0(self, X, y_h, grad=None):
        [x0, x0_h, y] = self.coarse_data(X, y_h)
        coarse_iter_class = self.params.coarse_iterator()

        if self.par_f.scale_coherent_gradient_init() is True:
            if grad is None:
                print(f"coherence: lv{self.par_f.level}")
                grad_x0 = self.grad(x0_h, y_h, self.ph_f, self.gfine, self.par_f)
            else:
                grad_x0 = grad(x0_h)

            v = self.projection(grad_x0)
            v -= self.grad(x0, y, self.physics, self.g, self.params)

            # Coarse gradient (first order coherent)
            grad_coarse = lambda x: self.grad(x, y, self.physics, self.g, self.params) + v
            level_iteration = coarse_iter_class(coarse_correction=v)
        else:
            grad_coarse = lambda x: self.grad(x, y, self.physics, self.g, self.params)
            level_iteration = coarse_iter_class()

        if self.is_large(x0):
            if self.params.level > 1:
                model = CoarseModel(self.g, self.f, self.physics, self.params)
                x0 = model.init_ml_x0({'est': [x0]}, y, grad=grad_coarse)
        else:
            print(f"Warning: Coarse init: image is small, cannot iterate below level {self.params.level}")

        f_init = lambda def_y, def_ph: {'est': [x0], 'cost': None}
        #iteration = GDIteration(has_cost=False)
        #iteration_class = self.pc.coarse_iterator()
        #iteration = iteration_class()

        model = optim.optim_builder(
            level_iteration,
            data_fidelity=self.f,
            prior=self.g,
            custom_init=f_init,
            max_iter=self.params.iters_init(),
            params_algo=self.params.params,
        )
        x_est_coarse = model(y, self.physics)
        return self.prolongation(x_est_coarse)

    def forward(self, X, y_h, grad=None):
        [x0, x0_h, y] = self.coarse_data(X, y_h)
        coarse_iter_class = self.params.coarse_iterator()
        #from torchvision.utils import save_image
        #save_image(
        #    #torch.norm(y[0, ::], dim=0, keepdim=True),
        #    y[0, ::], os.path.join(get_out_dir(), f"y_coarse{self.par.level}.png")
        #)

        if self.par_f.scale_coherent_gradient() is True:
            if grad is None:
                #print(f"coherence: lv{self.params.level} (fine)")
                grad_x0 = self.grad(x0_h, y_h, self.ph_f, self.gfine, self.par_f)
            else:
                grad_x0 = grad(x0_h)

            v = self.projection(grad_x0)
            v -= self.grad(x0, y, self.physics, self.g, self.params)

            # Coarse gradient (first order coherent)
            grad_coarse = lambda x: self.grad(x, y, self.physics, self.g, self.params) + v
            level_iteration = coarse_iter_class(coarse_correction=v)
        else:
            grad_coarse = lambda x: self.grad(x, y, self.physics, self.g, self.params)
            level_iteration = coarse_iter_class()

        if self.is_large(x0):
            if self.params.level > 1:
                model = CoarseModel(self.g, self.f, self.physics, self.params)
                diff = model({'est': [x0]}, y, grad=grad_coarse)
                x1 = x0 + self.params.stepsize() * diff
            else:
                x1 = x0
        else:
            print(f"Warning: Coarse model: image is small, cannot iterate below level {self.params.level}")
            x1 = x0

        if isinstance(self.g, CustTV):
            self.g.gamma_moreau = self.params.gamma_moreau()  # compute gradient on Moreau envelope

        f_init = lambda def_y, def_ph: {'est': [x1], 'cost': None}
        model = optim.optim_builder(
            level_iteration,
            data_fidelity=self.f,
            prior=self.g,
            custom_init=f_init,
            max_iter=self.params.iters(),
            params_algo=self.params.params,
        )
        x_est_coarse = model(y, self.physics)
        #save_image(
        #    #torch.norm(y[0, ::], dim=0, keepdim=True),
        #    x_est_coarse[0, ::], os.path.join(get_out_dir(), f"x_coarse{self.par.level}.png")
        #)

        assert not torch.isnan(x_est_coarse).any()

        return self.prolongation(x_est_coarse - x0) # / factor**2

