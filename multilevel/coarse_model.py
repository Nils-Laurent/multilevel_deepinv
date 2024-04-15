import torch
from deepinv.optim.optim_iterators import GDIteration
from deepinv.physics import Inpainting, Blur
import deepinv.optim as optim

# multilevel imports
from multilevel.info_transfer import DownsamplingTransfer
from multilevel.coarse_gradient_descent import CGDIteration
import multilevel.iterator as multi_level


class CoarseModel(torch.nn.Module):
    def __init__(self, prior, data_fidelity, fine_physics, ml_params, *args, **kwargs):
        """
        :param multi_level.MultiLevelParams ml_params: all parameters
        """
        super().__init__(*args, **kwargs)
        self.g = prior
        self.f = data_fidelity
        self.fph = fine_physics
        self.physics = None
        self.ph = ml_params
        self.pc = self.ph.coarse_params()
        self.cit_str = self.ph.cit()
        self.cit_op = None

    def projection(self, x):
        if self.cit_op is None:
            self.cit_op = DownsamplingTransfer(x)
            if self.physics is None:
                self.coarse_physics()
        x_proj = self.cit_op.projection(x)
        return x_proj

    def prolongation(self, x):
        if self.cit_op is None:
            self.cit_op = DownsamplingTransfer(x)
        x_prol = self.cit_op.prolongation(x)
        return x_prol

    def grad(self, x, y, physics, params):
        grad_f = self.f.grad(x, y, physics)

        if hasattr(self.g, 'denoiser'):
            grad_g = self.g.grad(x, sigma_denoiser=params.g_param())
        elif hasattr(self.g, 'moreau_grad'):
            grad_g = self.g.moreau_grad(x, gamma=params.gamma_moreau())
        else:
            raise NotImplementedError("Gradient not defined in this case")

        return grad_f + params.lambda_r() * grad_g

    def coarse_data(self, X, y_h):
        x0_h = X['est']
        if not isinstance(x0_h, torch.Tensor):
            x0_h = x0_h[0]  # primal value of 'est'

        # Projection
        x0 = self.projection(x0_h)
        y = self.projection(y_h)

        return x0, x0_h, y

    def coarse_physics(self):
        if isinstance(self.fph, Inpainting):
            m_fine = self.fph.mask.data
            m_coarse = self.projection(m_fine)
            c_mask = torch.squeeze(m_coarse, 0)
            self.physics = Inpainting(tensor_size=m_coarse.shape, mask=c_mask)
        elif isinstance(self.fph, Blur):
            fph = self.fph
            if fph.filter.shape[2] < 4 or fph.filter.shape[3] < 4 :
                filt = fph.filter
            else:
                filt = self.projection(fph.filter)
            self.physics = Blur(filter=filt, padding=fph.padding, device=fph.device)
        else:
            raise NotImplementedError("Coarse physics not implemented for " + str(type(self.fph)))

        return self.physics

    def init_ml_x0(self, X, y_h):
        [x0, x0_h, y] = self.coarse_data(X, y_h)

        if self.ph.level > 1:
            model = CoarseModel(self.g, self.f, self.physics, self.pc)
            x0 = model.init_ml_x0({'est': [x0]}, y)

        f_init = lambda def_y, def_ph: {'est': [x0], 'cost': None}
        iteration = GDIteration(has_cost=False)
        model = optim.optim_builder(
            iteration,
            data_fidelity=self.f,
            prior=self.g,
            custom_init=f_init,
            max_iter=self.pc.iters(),
            params_algo=self.pc.params,
        )
        x_est_coarse = model(y, self.physics)
        return self.prolongation(x_est_coarse)

    def forward(self, X, y_h, grad=None):
        [x0, x0_h, y] = self.coarse_data(X, y_h)

        if self.ph.scale_coherent_gradient() is True:
            if grad is None:
                grad_x0 = self.grad(x0_h, y_h, self.fph, self.ph)
            else:
                grad_x0 = grad(x0_h)

            v = self.projection(grad_x0)
            v -= self.grad(x0, y, self.physics, self.pc)

            # Coarse gradient (first order coherent)
            grad_coarse = lambda x: self.grad(x, y, self.physics, self.pc) + v
        else:
            grad_coarse = lambda x: self.grad(x, y, self.physics, self.pc)

        level_iteration = CGDIteration(has_cost=False, coherent_grad_fn=grad_coarse)
        iteration = multi_level.MultiLevelIteration(level_iteration, grad_fn=grad_coarse, has_cost=False)

        f_init = lambda def_y, def_ph: {'est': [x0], 'cost': None}
        model = optim.optim_builder(
            iteration,
            data_fidelity=self.f,
            prior=self.g,
            custom_init=f_init,
            max_iter=self.pc.iters(),
            params_algo=self.pc.params,
        )
        x_est_coarse = model(y, self.physics)

        return self.prolongation(x_est_coarse - x0)

