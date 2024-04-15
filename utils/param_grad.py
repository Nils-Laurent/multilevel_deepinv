import deepinv
import torch
from deepinv import train
from deepinv.physics import GaussianNoise

from multilevel.info_transfer import BlackmannHarris
from tests.utils import physics_from_exp, data_from_user_input, single_level_params, count_parameters
from tests.utils import standard_multilevel_param
from tests.test_alg import RunAlgorithm

def tune_param(data_in, param_exp, device, max_lv):
    #tune_param_red(data_in, param_exp, device, max_lv)
    tune_param_tv(data_in, param_exp, device, max_lv)

def tune_param_red(data_in, params_exp, device, max_lv):
    lambda_red = 0.2
    g_param = 0.05
    noise_pow = params_exp["noise_pow"]

    g = GaussianNoise(sigma=noise_pow)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    lip_g = 300
    iters_fine = 10
    iters_vec = [3, 2, 2, iters_fine]
    if device == "cpu":
        iters_vec = [5, 5, iters_fine]
    if max_lv < len(iters_vec):
        iters_vec = iters_vec[-max_lv:]

    levels = len(iters_vec)

    params_algo = {
        'cit': BlackmannHarris(),
        'iml_max_iter': 8,
        'scale_coherent_grad': True
    }

    p_red = params_algo.copy()
    p_red = standard_multilevel_param(p_red, it_vec=iters_vec)
    p_red['g_param'] = g_param
    p_red['lip_g'] = lip_g  # denoiser Lipschitz constant
    p_red['lambda'] = lambda_red
    p_red['step_coeff'] = 0.9  # no convex setting
    p_red['stepsize'] = p_red['step_coeff'] / (1.0 + lambda_red * lip_g)
    param_train = ['lambda', 'g_param']
    param_train = ['lambda']

    param_init = {'init_ml_x0': [5] * len(iters_vec)}
    ra = RunAlgorithm(data, physics, params_exp, device, param_init=param_init, r_model=True, trainable_params=param_train)
    model = ra.RED_GD(p_red)

    # todo : split learning_rate for lambda and sigma_denoiser
    #real_name = []
    #for p in param_train:
    #    real_name.append(f"init_params_algo.{p}.0")
    #d0 = {}
    #for kv in model.named_parameters():
    #    d0[kv[0]]= kv[1]
    #named_param_train_model = list(filter(lambda kv: kv[0] in real_name, model.named_parameters()))
    #param_train_model = [value0 for name0, value0 in named_param_train_model]
    #lr0 = 1e-4
    #[{'params': model.parameters(), 'lr':lr0}, {'params': model.parameters()}]

    n_epochs = 10
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    tune_losses = [deepinv.loss.SupLoss(metric=deepinv.metric.mse())]
    print(f"lr = {learning_rate}")
    step_size = int(n_epochs * 0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print(f"step_size = {step_size}")

    c, kvp = count_parameters(model, pr=True)
    for k, v in kvp.items():
        print(f"{k} = {v.item()}")

    model = train(
        model=model,
        train_dataloader=data,
        # scheduler = scheduler,
        epochs=n_epochs,
        losses=tune_losses,
        physics=physics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    c, kvp = count_parameters(model, pr=False)
    for k, v in kvp.items():
        print(f"{k} = {v.item()}")

def tune_param_tv(data_in, params_exp, device, max_lv):
    lambda_tv = 1.0
    noise_pow = params_exp["noise_pow"]

    g = GaussianNoise(sigma=noise_pow)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    iters_fine = 2
    #if device == "cpu":

    params_algo = {
        'cit': BlackmannHarris(),
        'iml_max_iter': 8,
        'scale_coherent_grad': True
    }

    p_tv = params_algo.copy()
    p_tv['iters'] = iters_fine
    p_tv['prox_crit'] = 1e-6
    p_tv['prox_max_it'] = 1000
    p_tv['prox_max_it'] = 2
    p_tv['g_param'] = 1.0
    p_tv['lip_g'] = 1.0
    p_tv['lambda'] = lambda_tv
    p_tv['step_coeff'] = 1.9  # no convex setting
    p_tv['stepsize'] = p_tv['step_coeff'] / (1.0 + lambda_tv)
    param_train = ['lambda']

    ra = RunAlgorithm(data, physics, params_exp, device=device, r_model=True, trainable_params=param_train)
    model = ra.TV_PGD(p_tv, use_cost=False)
    #ra = RunAlgorithm(data, physics, params_exp, device, r_model=True, trainable_params=param_train)
    #model = ra.TV_PGD(p_tv)

    n_epochs = 10
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    tune_losses = [deepinv.loss.SupLoss(metric=deepinv.metric.mse())]
    print(f"lr = {learning_rate}")
    step_size = int(n_epochs * 0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print(f"step_size = {step_size}")

    c, kvp = count_parameters(model, pr=True)
    for k, v in kvp.items():
        print(f"{k} = {v.item()}")

    model = train(
        model=model,
        train_dataloader=data,
        # scheduler = scheduler,
        epochs=n_epochs,
        losses=tune_losses,
        physics=physics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    c, kvp = count_parameters(model, pr=False)
    for k, v in kvp.items():
        print(f"{k} = {v.item()}")