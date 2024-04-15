import torch
from deepinv.optim.optim_iterators.optim_iterator import OptimIterator
from multilevel.coarse_model import CoarseModel


class MultiLevelIteration(OptimIterator):
    def __init__(self, fine_iteration, grad_fn = None, **kwargs):
        super(MultiLevelIteration, self).__init__(**kwargs)
        self.fine_iteration=fine_iteration
        self.F_fn = fine_iteration.F_fn
        self.has_cost = fine_iteration.has_cost
        self.grad_fn = grad_fn
        self.cur_ml_iter = 0

    def multilevel_step(self, X, data_fidelity, prior, params, y, physics):
        ml_params = MultiLevelParams(params)
        if ml_params.level == 1 or self.cur_ml_iter >= ml_params.iml_max_iter():
            return X

        self.cur_ml_iter += 1
        model = CoarseModel(prior, data_fidelity, physics, ml_params)
        diff = model(X, y, grad=self.grad_fn)
        step = 1.0

        if self.fine_iteration.has_cost:
            # backtracking
            def cost_fn(x):
                return self.F_fn(x, data_fidelity, prior, params, y, physics)

            x0 = X['est'][0]
            nb = 0
            while cost_fn(x0 + step * diff) > cost_fn(x0):
                step = step / 2
                nb += 1
        x_bt = X['est'][0] + step * diff
        Y = {'est': [x_bt]}

        return Y

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        Y = self.multilevel_step(X, cur_data_fidelity, cur_prior, cur_params, y, physics)

        X2 = self.fine_iteration(Y, cur_data_fidelity, cur_prior, cur_params, y, physics)
        return X2


class MultiLevelParams:
    def __init__(self, params):
        self.params = params
        self.n_level = self._get_scalar_init('n_levels', params)
        self.level = self._get_scalar_init('level', params)
        self.g_lipschitz = self._get_scalar_init('lip_g', params)
        self.f_lipschitz = 1.0

    def coarse_params(self):
        if self.level == 0:
            raise ValueError("Cannot get coarser params")
        cp_params = self.params.copy()
        cp_params['level'] = self.level - 1
        cp = MultiLevelParams(cp_params)
        cp._set_coarse()
        return cp

    def g_param(self):
        return self._get_scalar('g_param')

    def cit(self):
        return self._get_scalar('cit')

    def scale_coherent_gradient(self):
        return self._get_scalar('scale_coherent_grad')

    def iml_max_iter(self):
        return self._get_scalar('iml_max_iter')

    def stepsize(self):
        return self._get_scalar('stepsize')

    def lambda_r(self):
        return self._get_scalar('lambda')

    # ============================== MULTILEVEL ==============================

    def gamma_moreau(self):
        return self._get_from_ml('gamma_moreau')

    def iters(self):
        return self._get_from_ml('iters')

    # ========================== INTERNAL FUNCTIONS ==========================
    # internal functions
    def _get_scalar(self, key):
        return self._get_scalar_init(key, self.params)

    @staticmethod
    def _get_scalar_init(key, params):
        scalar = params[key]
        if isinstance(scalar, torch.Tensor):
            return scalar
        if isinstance(scalar, list):
            scalar = scalar[0]
        return scalar

    def _get_from_ml(self, key):
        ml_dict = self.params['params_multilevel']
        if isinstance(ml_dict, list):
            ml_dict = self.params['params_multilevel'][0]
        elif not isinstance(ml_dict, dict):
            raise TypeError('params_multilevel must be a list or dict')

        return ml_dict[key][self.level - 1]

    def _set_coarse(self):
        self.params['lambda'] = self.lambda_r() / 4
        step_coeff = self._get_scalar('step_coeff')
        if 'gamma_moreau' in self.params.keys():
            self.params['stepsize'] = step_coeff / (self.g_lipschitz * self.gamma_moreau() + self.f_lipschitz)
        else:
            self.params['stepsize'] = step_coeff / (self.g_lipschitz * self.lambda_r() + self.f_lipschitz)

        if isinstance(self.params['params_multilevel'], dict):
            self.params['params_multilevel'] = [self.params['params_multilevel']]

        for key, value in self.params.items():
            if (isinstance(value, torch.nn.parameter.Parameter) or
                isinstance(value, torch.Tensor)):
                self.params[key] = [value]


