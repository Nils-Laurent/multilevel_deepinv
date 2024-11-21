import os
from os.path import join

import deepinv.loss
import torch
from deepinv.optim import optim_builder
from deepinv.physics import MRI
from torchvision.utils import save_image

from multilevel_utils.complex_denoiser import to_complex_denoiser
from utils.ml_dataclass import *
from utils.ml_dataclass_nonexp import *
from utils.ml_dataclass_exp import *
from deepinv.optim.dpir import get_DPIR_params
from deepinv.models import DRUNet
from deepinv.optim.optim_iterators import GDIteration, PGDIteration
from deepinv.optim.prior import PnP
from deepinv.utils.logger import MetricLogger
from torch.utils.data import DataLoader

# multilevel imports
from multilevel.prior import TVPrior
import multilevel
from multilevel.iterator import MultiLevelIteration, MultiLevelParams
from multilevel.coarse_model import CoarseModel
from multilevel_utils.radon import Tomography
from tests.parameters import single_level_params

from utils.mat_utils import gen_matlab_conf, gen_mat_cost, gen_mat_dataset_psnr
from utils.paths import gen_fname, get_out_dir


class RunAlgorithm:
    def __init__(
        self,
        data,
        physics,
        params_exp,
        device,
        #param_init=None,
        r_model=False,
        trainable_params=None,
        return_timer=False,
        def_name=None,
    ):
        self.data = data
        self.physics = physics
        self.params_exp = params_exp
        #self.data_fidelity = L2()
        self.data_fidelity = ConfParam().data_fidelity()
        self.device = device
        self.ret_model = r_model
        self.verbose = False
        self.time_iter = return_timer
        self.x0 = None

        self.trainable_params = trainable_params
        if trainable_params is None:
            self.trainable_params = []

        b = ("manual_seed" in params_exp.keys()) and (params_exp["manual_seed"] is True)
        self.manual_seed = b
        self.alg_name = def_name

        self.param_init = None

        self.is_gridsearch = False
        if 'gridsearch' in params_exp.keys():
            self.is_gridsearch = True

    def set_init(self, param_init):
        self.param_init = param_init

    def run_algorithm(self, m_class, params_algo):
        if hasattr(m_class, "edit_fn"):
            for fn in m_class.edit_fn:
                params_algo = fn(params_algo, self.params_exp)
                if self.param_init is not None:
                    # edit : prior Moreau, Student, NoReg
                    self.param_init = fn(self.param_init, self.params_exp)

        # set single level parameters
        if m_class in [MFb, MRed, MRedInit, MPnP, MPnPInit, MPnPProx, MPnPProxInit]\
                or hasattr(m_class, "single_level"):
            params_algo = single_level_params(params_algo)

        if "RED" in m_class().key:
            return self.RED_GD(params_algo)
        elif m_class in [MFb, MFbMLProx, MFbMLGD]:
            return self.TV_PGD(params_algo, use_cost=True)
        elif m_class in [
            MPnP, MPnPInit, MPnPML, MPnPMLInit, MPnPMLNoR, MPnPMLStud, MPnPMLStudInit,
            MPnPMoreau, MPnPMoreauInit, MPnPMLStudNoR, MPnPMLStudNoRInit,

            MPnPProx, MPnPProxInit, MPnPProxML, MPnPProxMLInit, MPnPProxMLStud, MPnPProxMLStudInit,
            MPnPProxMoreau, MPnPProxMoreauInit, MPnPProxMLStudNoR, MPnPProxMLStudNoRInit,

            MPnPNE, MPnPNEInit, MPnPNEML, MPnPNEMLInit, MPnPNEMLStud, MPnPNEMLStudInit,
            MPnPNEMoreau, MPnPNEMoreauInit,
        ]:
            return self.PnP_PGD(params_algo, use_cost=False)
        elif m_class in [MPnPML]:
            return self.PnP(params_algo)
        elif m_class in [MDPIRLong]:
            return self.DPIR(params_algo, def_iter=200)
        elif m_class in [MDPIR]:
            return self.DPIR(params_algo)
        else:
            raise NotImplementedError("Unrecognized model {}".format(m_class))

    def RED_GD(self, params_algo):
        alg_name = "RED_GD"
        #net = DRUNet(pretrained="download", device=self.device)
        #denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        #prior = RED(denoiser)
        prior = params_algo['prior']
        iteration = GDIteration(has_cost=False)
        if 'level' in params_algo.keys() and params_algo['level'] > 1:
            iteration = MultiLevelIteration(iteration)

        return self._run_algorithm(iteration, prior, params_algo, alg_name)

    def PnP(self, params_algo):
        alg_name = "PnP"
        prior = params_algo['prior']
        #denoiser = DRUNet(pretrained="download", device=self.device)
        #prior = PnP(denoiser)

        iteration = PGDIteration()
        if 'level' in params_algo.keys() and params_algo['level'] > 1:
            iteration = MultiLevelIteration(iteration)
        return self._run_algorithm(iteration, prior, params_algo, alg_name)

    def PnP_PGD(self, params_algo, use_cost=True):
        alg_name = "PnP_PGD"
        prior = params_algo['prior']
        #denoiser = GSDRUNet(pretrained="download", train=False, device=self.device)
        #prior = PnP(denoiser)

        def F_fn(x, data_fidelity, prior, cur_params, y, physics):
            denoiser = prior.denoiser
            prior_value = denoiser.potential(x, cur_params["g_param"], reduce=False)  # should be phi_sigma
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
        return self._run_algorithm(iteration, prior, params_algo, alg_name)

    def TV_PGD(self, params_algo, use_cost=True):
        alg_name = "TV_PGD"
        prior = multilevel.prior.TVPrior(def_crit=params_algo["prox_crit"], n_it_max=params_algo["prox_max_it"])

        def F_fn(x, data_fidelity, prior, cur_params, y, physics):
            prior_value = prior(x, cur_params["g_param"], reduce=False)  # g_param ?
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

        return self._run_algorithm(iteration, prior, params_algo, alg_name)

    def DPIR(self, params_algo, def_iter=8):
        alg_name = "DPIR"
        sigma_denoiser, stepsize, max_iter = get_DPIR_params(self.params_exp['noise_pow'], def_iter)
        params_algo['stepsize'] = stepsize
        params_algo['g_param'] = sigma_denoiser
        params_algo['iters'] = max_iter

        # Specify the denoising prior
        if self.params_exp['problem'] == "mri":
            denoiser = DRUNet(
                in_channels=ConfParam().denoiser_in_channels,
                out_channels=ConfParam().denoiser_in_channels,
                pretrained="download", device=self.device)
            denoiser = to_complex_denoiser(denoiser, mode="separated")
            prior = PnP(denoiser=denoiser)
        else:
            prior = PnP(denoiser=DRUNet(pretrained="download", device=self.device))
        iteration = 'HQS'
        return self._run_algorithm(iteration, prior, params_algo, alg_name)

    def _run_algorithm(self, iteration, prior, params_algo, alg_name_):
        alg_name = self.alg_name
        if self.manual_seed is True:
            print("Using torch.manual_seed(0)")
            torch.manual_seed(0)

        f_init = False

        params_init = self.param_init
        if params_init is not None:
            ml_params = MultiLevelParams(params_init)
            #params_algo_init = params_algo.copy()
            #ml_params = MultiLevelParams(params_algo_init)

            def use_adjoint(physics):
                if isinstance(physics, Tomography):
                    return True
                elif isinstance(physics, MRI):
                    return True

            def init_ml_x0(y, physics, F_fn=None):
                cm = CoarseModel(prior, self.data_fidelity, physics, ml_params)
                if use_adjoint(physics):
                    x0 = cm.init_ml_x0({'est': [physics.A_adjoint(y)]}, y)
                else:
                    x0 = cm.init_ml_x0({'est': [y]}, y)

                self.x0 = x0

                if F_fn is None:
                    return {'est': [x0]}

                cost = F_fn(x0, self.data_fidelity, prior, ml_params, y, physics)
                return {'est': [x0], 'cost': cost}

            f_init = init_ml_x0

            # ===== GROUND TRUTH INIT FOR TESTING PURPOSE =====
            ground_truth_init = False
            if ground_truth_init is True:
                def init_gt(y, physics, F_fn=None):
                    return {'est': [self.data], 'cost': None}
                f_init = init_gt

        if isinstance(iteration, MultiLevelIteration):
            iters = params_algo['params_multilevel'][0]['iters'][-1]
        else:
            iters = params_algo['iters']

        #model = unfolded_builder(
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
            params_algo=params_algo,
            custom_init=f_init,
            #trainable_params=self.trainable_params,
            #device=self.device,
        )

        if self.ret_model:
            return model

        if not self.is_gridsearch:
            print("run", alg_name)

        if isinstance(self.data, DataLoader):
            m = MetricLogger()
            online = self.params_exp['online']
            model.eval()
            progress_bar = not self.is_gridsearch

            dinv_res = deepinv.test(
                model, self.data, self.physics,
                device=self.device, online_measurements=online, metric_logger=m, time_iter=self.time_iter,
                show_progress_bar=progress_bar
            )

            if self.is_gridsearch:
                return dinv_res['PSNR']

            test_psnr, test_std_psnr, init_psnr, init_std_psnr = dinv_res
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

            #model(y, self.physics)
            #return None
            #cProfile.runctx(
            #    'model(y, ph)',
            #    {'model': model, 'y': y, 'ph': self.physics},
            #    {},
            #    filename='runctx_multilevel'
            #)
            x_est, met = model(y, self.physics, x_gt=x_ref, compute_metrics=True, time_iter=self.time_iter)

            #loss = deepinv.loss.PSNR(max_pixel=1)
            #print("loss(x_est_0, truth_0) =", loss(x_est[:, 0:1, ::], x_ref[:, 0:1, ::]))
            #print("loss(x_est_1, truth_1) =", loss(x_est[:, 1:2, ::], x_ref[:, 1:2, ::]))
            #print("loss(x_est, truth) =", loss(x_est, x_ref))
            #return None

            # ==================== Save results ====================
            print("saving:", f_prefix)

            p_exp = self.params_exp
            #exp_prefix = f"{p_exp['img_name']}_n{p_exp['noise_pow']}_{p_exp['problem']}"

            x0_disp = self.x0
            x_disp = x_ref
            y_disp = y
            xe_disp = x_est

            vmax = torch.max(x_disp)
            img_prefix, exp_prefix = gen_fname(params_algo, p_exp, alg_name)

            if p_exp['problem'] == 'mri':
                if self.x0 is not None:
                    x0_disp = torch.norm(self.x0, dim=1, p=2, keepdim=True)
                x_disp = torch.norm(x_ref, dim=1, p=2, keepdim=True)
                y_disp = torch.norm(y, dim=1, p=2, keepdim=True)
                xe_disp = torch.norm(x_est, dim=1, p=2, keepdim=True)

                x_lin = self.physics.A_adjoint(y)
                x_lin_disp = torch.norm(x_lin, dim=1, p=2, keepdim=True)

                img_name = join(get_out_dir(), exp_prefix + "_xlin.png")
                save_image(x_lin_disp[0, ::]/vmax, img_name)

            if False and self.x0 is not None:
                save_image(
                    x0_disp[0, ::],
                    os.path.join(get_out_dir(), f"{img_prefix}_x0.png")
                )
                x0n_disp = x0_disp / torch.max(x0_disp)
                save_image(
                    x0n_disp[0, ::],
                    os.path.join(get_out_dir(), f"{img_prefix}_x0n.png")
                )
                x0diff = x_disp[0, ::] - x0n_disp[0, ::]
                save_image(
                    x0diff/torch.max(x0diff),
                    os.path.join(get_out_dir(), f"{img_prefix}_x0diffn.png")
                )

            img_name = join(get_out_dir(), exp_prefix + "_x_truth.png")
            save_image(x_disp[0, ::]/vmax, img_name)

            img_name = join(get_out_dir(), exp_prefix + "_y.png")
            save_image(y_disp[0, ::]/vmax, img_name)

            img_name = join(get_out_dir(), img_prefix + ".png")
            save_image(xe_disp[0, ::]/vmax, img_name)

            if p_exp['problem'] == 'mri':
                img_name = join(get_out_dir(), exp_prefix + "_mask0.png")
                save_image(self.physics.mask[0, 0, ::].unsqueeze(0), img_name)
                img_name = join(get_out_dir(), exp_prefix + "_mask1.png")
                save_image(self.physics.mask[0, 1, ::].unsqueeze(0), img_name)


            print("PSNR: ", met['psnr'][0][-1])
            print('--')

            dict_metrics = met
            gen_mat_cost(dict_metrics, f_prefix, params_algo)

            return met
