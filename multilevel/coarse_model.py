import torch
from deepinv.optim.optim_iterators import GDIteration
from deepinv.physics import Inpainting, Blur
import deepinv.optim as optim

# multilevel imports
from multilevel.info_transfer import DownsamplingTransfer
from multilevel.coarse_gradient_descent import CGDIteration
import multilevel.iterator as multi_level


class CoarseModel(torch.nn.Module):
    def __init__(self, prior, data_fidelity, fine_physics, params_ml, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = prior
        self.f = data_fidelity
        self.fph = fine_physics
        self.physics = None
        self.cit_str = params_ml['cit']
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
            grad_g = self.g.grad(x, sigma_denoiser=params['g_param'])
        elif hasattr(self.g, 'moreau_grad') and 'gamma_moreau' in params.keys():
            grad_g = self.g.moreau_grad(x, gamma=params['gamma_moreau'])
        else:
            grad_g = self.g.grad(x)

        return grad_f + params['lambda'] * grad_g

    def coarse_data(self, X, y_h, params_ml_h):
        params_ml = params_ml_h.copy()
        params_ml['level'] = params_ml['level'] - 1

        params = multi_level.MultiLevelIteration.get_level_params(params_ml)

        # todo: compute lipschitz constant in a clever way
        if 'gamma_moreau' in params.keys():
            f_lipschitz = 1.0
            params['stepsize'] = 1.0 / (f_lipschitz + params['gamma_moreau'])

        x0_h = X['est']
        if not isinstance(x0_h, torch.Tensor):
            x0_h = x0_h[0]  # primal value of 'est'

        # Projection
        x0 = self.projection(x0_h)
        y = self.projection(y_h)

        return x0, x0_h, y, params

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

    def init_ml_x0(self, X, y_h, params_ml_h):
        [x0, x0_h, y, params] = self.coarse_data(X, y_h, params_ml_h)

        if params['level'] > 1:
            model = CoarseModel(self.g, self.f, self.physics, params)
            x0 = model.init_ml_x0({'est': [x0]}, y, params)

        f_init = lambda def_y, def_ph: {'est': [x0], 'cost': None}
        iters_vec = params['params_multilevel'].params['iters']
        iteration = GDIteration(has_cost=False)
        model = optim.optim_builder(
            iteration,
            data_fidelity=self.f,
            prior=self.g,
            custom_init=f_init,
            max_iter=iters_vec[params['level'] - 1],
            params_algo=params.copy(),
        )
        x_est_coarse = model(y, self.physics)
        return self.prolongation(x_est_coarse)

    def forward(self, X, y_h, params_ml_h, grad=None):
        [x0, x0_h, y, params] = self.coarse_data(X, y_h, params_ml_h)
        params_h = multi_level.MultiLevelIteration.get_level_params(params_ml_h)

        if params['scale_coherent_grad'] is True:
            if grad is None:
                grad_x0 = self.grad(x0_h, y_h, self.fph, params_h)
            else:
                grad_x0 = grad(x0_h)

            v = self.projection(grad_x0)
            v -= self.grad(x0, y, self.physics, params)

            # Coarse gradient (first order coherent)
            grad_coarse = lambda x: self.grad(x, y, self.physics, params) + v
        else:
            grad_coarse = lambda x: self.grad(x, y, self.physics, params)

        level_iteration = CGDIteration(has_cost=False, coherent_grad_fn=grad_coarse)
        iteration = multi_level.MultiLevelIteration(level_iteration, grad_fn=grad_coarse, has_cost=False)

        f_init = lambda def_y, def_ph: {'est': [x0], 'cost': None}
        iters_vec = params['params_multilevel'].params['iters']
        model = optim.optim_builder(
            iteration,
            data_fidelity=self.f,
            prior=self.g,
            custom_init=f_init,
            max_iter=iters_vec[params['level'] - 1],
            params_algo=params.copy(),
        )
        x_est_coarse = model(y, self.physics)

        return self.prolongation(x_est_coarse - x0)
