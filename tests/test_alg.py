import copy
import torch

from deepinv.optim.dpir import get_DPIR_params
from deepinv.utils import plot, plot_curves
from deepinv.models import DRUNet
from deepinv.optim import PnP, L2, optim_builder
from deepinv.optim.optim_iterators import GDIteration, PGDIteration
from deepinv.optim.prior import ScorePrior

# multilevel imports
from optim.prior import TVPrior
from optim.optim_iterators.multi_level import MultiLevelIteration
from optim.coarse_model import CoarseModel

from utils.gen_mat import gen_matlab_conf, gen_mat_cost, gen_mat_images
from utils.paths import gen_fname


class RunAlgorithm:
    def __init__(self, data, physics, params_exp, device, param_init=None):
        self.data = data
        self.physics = physics
        self.params_exp = params_exp
        self.data_fidelity = L2()
        self.device = device

        if param_init is None:
            param_init = {}
        self.param_init = param_init

    def PNP_PGD(self, params_algo):
        alg_name = "PNP_PGD"
        denoiser = DRUNet(pretrained="download", train=False, device=self.device)
        prior = PnP(denoiser)
        iteration = PGDIteration(has_cost=False)
        if 'level' in params_algo.keys() and params_algo['level'] > 1:
            raise NotImplementedError("PnP ML not yet implem.")

        return self.run_algorithm(iteration, prior, params_algo, alg_name)

    def RED_GD(self, params_algo):
        alg_name = "RED_GD"
        denoiser = DRUNet(pretrained="download", train=False, device=self.device)
        prior = ScorePrior(denoiser)
        iteration = GDIteration(has_cost=False)
        if 'level' in params_algo.keys() and params_algo['level'] > 1:
            iteration = MultiLevelIteration(iteration)

        return self.run_algorithm(iteration, prior, params_algo, alg_name)

    def TV_PGD(self, params_algo):
        alg_name = "TV_PGD"
        prior = TVPrior(def_crit=params_algo["prox_crit"], n_it_max=params_algo["prox_max_it"])

        def F_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics):
            return cur_data_fidelity(x, y, physics) + cur_params['lambda'] * cur_prior(x)

        iteration = PGDIteration(has_cost=True, F_fn=F_fn)
        if 'level' in params_algo.keys() and params_algo['level'] > 1:
            iteration = MultiLevelIteration(iteration)

        return self.run_algorithm(iteration, prior, params_algo, alg_name)

    def DPIR(self, params_algo):
        alg_name = "DPIR"
        sigma_denoiser, stepsize, max_iter = get_DPIR_params(self.params_exp['noise_pow'])
        params_algo['stepsize'] = stepsize
        params_algo['g_param'] = sigma_denoiser
        params_algo['iters'] = max_iter

        # Specify the denoising prior
        prior = PnP(denoiser=DRUNet(pretrained="download", train=False, device=self.device))
        iteration = 'HQS'
        return self.run_algorithm(iteration, prior, params_algo, alg_name)

    def run_algorithm(self, iteration, prior, params_algo_in, alg_name):
        params_algo = copy.deepcopy(params_algo_in)
        f_init = lambda x, physics: {'est': [x], 'cost': None}

        params_init = copy.deepcopy(self.param_init)
        if 'x0' in params_init.keys():
            x0 = params_init['x0']
            f_init = lambda x, physics: {'est': [x0], 'cost': None}
        elif 'init_ml_x0' in params_init:
            params_init['params_multilevel'].params['iters'] = params_init['init_ml_x0']
            params_init.pop('init_ml_x0', None)
            def init_ml_x0(x, physics):
                cm = CoarseModel(prior, self.data_fidelity, physics, params_init)
                x0 = cm.init_ml_x0({'est': [x]}, x, params_init)
                return {'est': [x0], 'cost': None}

            f_init = init_ml_x0

        if isinstance(iteration, MultiLevelIteration):
            iters = params_algo['params_multilevel'].params['iters'][-1]
        else:
            iters = params_algo['iters']

        model = optim_builder(
            iteration=iteration,
            prior=prior,
            data_fidelity=self.data_fidelity,
            max_iter=iters,
            g_first=False,
            early_stop=True,
            crit_conv='residual',
            thres_conv=1e-6,
            verbose=True,
            params_algo=copy.deepcopy(params_algo),
            custom_init=f_init
        )

        print("run", alg_name)

        if not isinstance(self.data, torch.Tensor):
            raise NotImplementedError("Not implemented yet")
        else:
            x_ref = self.data
            y = self.physics(x_ref)  # A(x) + noise
            x_est, met = model(y, self.physics, x_gt=x_ref, compute_metrics=True)

            # ==================== Save results ====================
            f_prefix, exp = gen_fname(params_algo, self.params_exp, alg_name)
            print("saving:", f_prefix)

            x0 = f_init(y, self.physics)['est'][0]
            dict_img = {
                "x_ref": x_ref,
                "y": y,
                "x0": x0,
                "x_est": x_est,
            }

            plot([x_est, x0, x_ref, y], titles=["est", "x0", "ref", "y"])
            plot_curves(met)
            print(met['psnr'][0][-1])

            dict_metrics = met

            gen_matlab_conf(exp)
            gen_mat_images(dict_img, f_prefix, params_algo)
            gen_mat_cost(dict_metrics, f_prefix, params_algo)

            return x_est, met
