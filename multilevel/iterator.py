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

    def ml_init(self, X, data_fidelity, prior, params, y, physics):
        ml_params = MultiLevelParams(params, params_init=True)
        model = CoarseModel(prior, data_fidelity, physics, ml_params)
        x0 = model.init_ml_x0(X, y)
        return {'est': [x0]}

    def multilevel_step(self, X, data_fidelity, prior, params, y, physics):
        ml_params = MultiLevelParams(params)
        if (ml_params.level == 1
                or not (ml_params.it_index() in ml_params.multilevel_indices())):
            return X

        model = CoarseModel(prior, data_fidelity, physics, ml_params)
        diff = model(X, y, grad=self.grad_fn)

        x0 = X['est'][0]
        if ml_params.backtracking() and self.fine_iteration.has_cost:
            # backtracking
            def cost_fn(x):
                return self.F_fn(x, data_fidelity, prior, params, y, physics)

            step = 1.0
            nb = 0
            while cost_fn(x0 + step * diff) > cost_fn(x0):
                step = step / 2
                nb += 1
            x_bt = x0 + step * diff
            Y = {'est': [x_bt], 'cost': cost_fn(x_bt)}
        else:
            Y = {'est': [x0 + diff]}

        return Y

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        # initialization block
        if cur_params['it_index'] == 0 and cur_params['ml_init'] is True:
            return self.ml_init(X, cur_data_fidelity, cur_prior, cur_params, y, physics)

        # ML block
        Y = self.multilevel_step(X, cur_data_fidelity, cur_prior, cur_params, y, physics)

        # fine level scheme block
        X2 = self.fine_iteration(Y, cur_data_fidelity, cur_prior, cur_params, y, physics)

        return X2


class MultiLevelParams:
    def __init__(self, params, params_init=False):
        self.params = params
        self.n_level = self._get_scalar_init('n_levels', params)
        self.level = self._get_scalar_init('level', params)
        if params_init is True:
            self.level = self._get_scalar_init('level_init', params)
        #self.g_lipschitz = self._get_scalar_init('lip_g', params)
        #self.f_lipschitz = 1.0

    def coarse_params(self):
        if self.level == 0:
            raise ValueError("Cannot get coarser params")
        cp_params = self.params.copy()  # top level only
        cp_params['level'] = self.level - 1
        cp = MultiLevelParams(cp_params)
        cp._set_coarse()
        return cp

    def g_param(self):
        return self._get_scalar('g_param')

    def cit(self):
        return self._get_scalar('cit')

    def scale_coherent_gradient_init(self):
        return self._get_bool('scale_coherent_grad_init')

    def scale_coherent_gradient(self):
        return self._get_bool('scale_coherent_grad')

    def iml_max_iter(self):
        return self._get_scalar('iml_max_iter')

    def it_index(self):
        return self._get_scalar('it_index')

    def backtracking(self):
        return self._get_bool('backtracking')

    def multilevel_indices(self):
        return self._get_list('multilevel_indices')

    def coarse_iterator(self):
        return self._get_class('coarse_iterator')

    def ml_denoiser(self):
        return self._get_with_default('ml_denoiser', False)

    def coarse_prior(self):
        return self._get_with_default('coarse_prior', False)

    def coherence_prior(self):
        return self._get_with_default('coherence_prior', False)

    # ============================== MULTILEVEL ==============================

    def gamma_moreau(self):
        return self._get_from_ml('gamma_moreau')

    def iters_init(self):
        return self._get_from_ml('iters_init')

    def iters(self):
        return self._get_from_ml('iters')

    def lambda_r(self):
        return self._get_from_ml('lambda')

    def stepsize(self):
        return self._get_from_ml('stepsize')

    # ========================== INTERNAL FUNCTIONS ==========================
    # internal functions
    def _get_bool(self, key):
        return self._get_scalar(key)

    def _get_scalar(self, key):
        return self._get_scalar_init(key, self.params)

    def _get_list(self, key):
        assert isinstance(self.params[key], list)
        return self.params[key]

    def _get_class(self, key):
        r_class = self.params[key]
        if isinstance(r_class, list):
            r_class = r_class[0]
        return r_class

    def _get_with_default(self, key, default):
        if key in self.params.keys():
            return self.params[key]
        return default

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

    def _ml_has_key(self, key):
        ml_dict = self.params['params_multilevel']
        if isinstance(ml_dict, list):
            ml_dict = self.params['params_multilevel'][0]
        if key in ml_dict.keys():
            return True
        return False

    def _set_coarse(self):
        self.params['stepsize'] = self._get_from_ml('stepsize')
        if self._ml_has_key('lambda'):
            self.params['lambda'] = self._get_from_ml('lambda')

        if isinstance(self.params['params_multilevel'], dict):
            self.params['params_multilevel'] = [self.params['params_multilevel']]

        for key, value in self.params.items():
            if (isinstance(value, torch.nn.parameter.Parameter) or
                isinstance(value, torch.Tensor)):
                self.params[key] = [value]
            if key == 'multilevel_indices' and len(value) > 1:
                self.params[key] = [value]


