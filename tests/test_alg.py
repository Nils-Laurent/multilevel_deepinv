import deepinv
import torch

from deepinv.optim.dpir import get_DPIR_params
from deepinv.unfolded import unfolded_builder
from deepinv.utils import plot, plot_curves
from deepinv.models import DRUNet, GSDRUNet
from multilevel_utils.fixed_gsdrunet import GSDRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optim_iterators import GDIteration, PGDIteration
from deepinv.optim.prior import ScorePrior, RED, PnP
from deepinv.utils.logger import MetricLogger
from torch.utils.data import DataLoader

# multilevel imports
from multilevel.prior import TVPrior
import multilevel
from multilevel.iterator import MultiLevelIteration, MultiLevelParams
from multilevel.coarse_model import CoarseModel
from multilevel_utils.radon import Tomography
from tests.parameters import standard_multilevel_param

from utils.mat_utils import gen_matlab_conf, gen_mat_cost, gen_mat_images, gen_mat_dataset_psnr
from utils.paths import gen_fname


class RunAlgorithm:
    def __init__(
        self,
        data,
        physics,
        params_exp,
        device,
        param_init=None,
        r_model=False,
        trainable_params=None,
        return_timer=False
    ):
        self.data = data
        self.physics = physics
        self.params_exp = params_exp
        self.data_fidelity = L2()
        self.device = device
        self.ret_model = r_model
        self.verbose = False
        self.time_iter = return_timer

        self.trainable_params = trainable_params
        if trainable_params is None:
            self.trainable_params = []

        if param_init is None:
            param_init = {}
        self.param_init = param_init

        b = ("manual_seed" in params_exp.keys()) and (params_exp["manual_seed"] is True)
        self.manual_seed = b

    def RED_GD(self, params_algo):
        alg_name = "RED_GD"
        net = DRUNet(pretrained="download", train=False, device=self.device)
        denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        prior = RED(denoiser)
        iteration = GDIteration(has_cost=False)
        if 'level' in params_algo.keys() and params_algo['level'] > 1:
            iteration = MultiLevelIteration(iteration)

        return self.run_algorithm(iteration, prior, params_algo, alg_name)

    def PnP_PGD(self, params_algo):
        alg_name = "PnP_PGD"
        denoiser = GSDRUNet(pretrained="download", train=False, device=self.device)
        prior = PnP(denoiser)
        iteration = PGDIteration(has_cost=False)
        if 'level' in params_algo.keys() and params_algo['level'] > 1:
            iteration = MultiLevelIteration(iteration)
        return self.run_algorithm(iteration, prior, params_algo, alg_name)

    def TV_PGD(self, params_algo, use_cost=True):
        alg_name = "TV_PGD"
        prior = multilevel.prior.TVPrior(def_crit=params_algo["prox_crit"], n_it_max=params_algo["prox_max_it"])

        def F_fn(x, data_fidelity, prior, cur_params, y, physics):
            prior_value = prior(x, cur_params["g_param"], reduce=False)
            if prior_value.dim() == 0:
                reg_value = cur_params["lambda"] * prior_value
            else:
                if isinstance(cur_params["lambda"], float):
                    reg_value = (cur_params["lambda"] * prior_value).sum()
                else:
                    reg_value = (
                        cur_params["lambda"].flatten() * prior_value.flatten()
                    ).sum()
            return data_fidelity(x, y, physics) + reg_value

        iteration = PGDIteration(has_cost=use_cost, F_fn=F_fn)
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

    def run_algorithm(self, iteration, prior, params_algo, alg_name):
        if self.manual_seed is True:
            torch.manual_seed(0)

        #if isinstance(self.physics, Tomography):
        #    f_init = lambda x, physics: {'est': [physics.A_adjoint(x)], 'cost': None}
        #else:
        #    f_init = lambda x, physics: {'est': [x], 'cost': None}
        f_init = False

        params_init = self.param_init
        if 'init_ml_x0' in params_init.keys():
            alg_name = alg_name + "_x0ML"
            params_algo_init = params_algo.copy()
            standard_multilevel_param(params_algo_init, it_vec=params_init['init_ml_x0'])
            ml_params = MultiLevelParams(params_algo_init)

            def init_ml_x0(y, physics, F_fn=None):
                cm = CoarseModel(prior, self.data_fidelity, physics, ml_params)
                if isinstance(physics, Tomography):
                    x0 = cm.init_ml_x0({'est': [physics.A_adjoint(y)]}, y)
                else:
                    x0 = cm.init_ml_x0({'est': [y]}, y)

                if F_fn is None:
                    return {'est': [x0]}

                cost = F_fn(x0, self.data_fidelity, prior, ml_params, y, physics)
                return {'est': [x0], 'cost': cost}

            f_init = init_ml_x0

        if isinstance(iteration, MultiLevelIteration):
            iters = params_algo['params_multilevel'][0]['iters'][-1]
        else:
            iters = params_algo['iters']

        model = unfolded_builder(
            iteration=iteration,
            prior=prior,
            data_fidelity=self.data_fidelity,
            max_iter=iters,
            g_first=False,
            early_stop=True,
            crit_conv='residual',
            thres_conv=1e-6,
            verbose=True,
            params_algo=params_algo,
            custom_init=f_init,
            trainable_params=self.trainable_params,
            device=self.device,
        )

        if self.ret_model:
            return model

        print("run", alg_name)

        if isinstance(self.data, DataLoader):
            m = MetricLogger()
            test_psnr, test_std_psnr, init_psnr, init_std_psnr = deepinv.test(
                model, self.data, self.physics, device=self.device, online_measurements=True, metric_logger=m
            )
            dict_res = {
                "test_psnr": test_psnr,
                "test_std_psnr": test_std_psnr,
                "init_psnr": init_psnr,
                "init_std_psnr": init_std_psnr,
            }

            f_prefix, exp = gen_fname(params_algo, self.params_exp, alg_name)
            gen_matlab_conf(exp)
            gen_mat_dataset_psnr(dict_res, f_prefix, params_algo, exp)

            if self.verbose is True:
                print("saving:", f_prefix)
                print(alg_name, ": test_psnr = ", test_psnr)
                print(alg_name, ": test_std_psnr = ", test_std_psnr)
                print(alg_name, ": init_psnr = ", init_psnr)
                print(alg_name, ": init_std_psnr = ", init_std_psnr)

            return m
        else:
            model.eval()
            # Assumes self.data is an image of the form torch.Tensor
            x_ref = self.data
            y = self.physics(x_ref)  # A(x) + noise
            f_prefix, exp = gen_fname(params_algo, self.params_exp, alg_name)

            x_est, met = model(y, self.physics, x_gt=x_ref, compute_metrics=True, time_iter=self.time_iter)

            # ==================== Save results ====================
            print("saving:", f_prefix)

            if not (f_init is False):
                x0 = f_init(y, self.physics)['est'][0]
            else:
                x0 = y

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

            return met
