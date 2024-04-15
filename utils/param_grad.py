import deepinv
import torch
from deepinv import train
from deepinv.physics import GaussianNoise

from multilevel.info_transfer import BlackmannHarris
from multilevel.iterator import MultiLevelParams
from tests.utils import physics_from_exp, data_from_user_input, single_level_params, count_parameters
from tests.utils import standard_multilevel_param
from tests.test_alg import RunAlgorithm


def tune_param(data_in, params_exp, device, max_lv):
    lambda_red = 0.5
    g_param = 0.03
    noise_pow = params_exp["noise_pow"]

    g = GaussianNoise(sigma=noise_pow)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    lip_d = 200
    iters_fine = 10
    iters_vec = [1, 1, 1, iters_fine]
    if device == "cpu":
        iters_vec = [5, 5, iters_fine]
    if max_lv < len(iters_vec):
        iters_vec = iters_vec[-max_lv:]

    levels = len(iters_vec)
    p_multilevel = MultiLevelParams({"iters": iters_vec})

    params_algo = {
        'cit': BlackmannHarris(),
        'level': levels,
        'params_multilevel': p_multilevel,
        'iml_max_iter': 8,
    }

    p_red = standard_multilevel_param(params_algo, lambda_red, step_coeff=0.9, lip_g=lip_d, it_vec=iters_vec)
    p_red['g_param'] = g_param
    p_red['scale_coherent_grad'] = True

    param_init = {
        'init_ml_x0': [80] * levels,
        'lambda': lambda_red,
        'step_coeff': 0.9,
        'lip_g': lip_d,
    }

    param_train = ['lambda', 'g_param']
    # ra = RunAlgorithm(data, physics, params_exp, device=device, r_model=True, trainable_params=param_train)
    # model = ra.RED_GD(single_level_params(p_red))
    ra = RunAlgorithm(data, physics, params_exp, device, param_init=param_init, r_model=True, trainable_params=param_train)
    model = ra.RED_GD(p_red)

    #real_name = []
    #for p in param_train:
    #    real_name.append(f"init_params_algo.{p}.0")
    #d0 = {}
    #for kv in model.named_parameters():
    #    d0[kv[0]]= kv[1]
    #named_param_train_model = list(filter(lambda kv: kv[0] in real_name, model.named_parameters()))
    #param_train_model = [value0 for name0, value0 in named_param_train_model]

    n_epochs = 10
    sgd_learning_rate = 1e-4
    #lr0 = 1e-4
    #[{'params': model.parameters(), 'lr':lr0}, {'params': model.parameters()}]
    optimizer = torch.optim.Adam(params=model.parameters(), lr=sgd_learning_rate)

    tune_losses = [deepinv.loss.SupLoss(metric=deepinv.metric.mse())]

    print(f"lr = {sgd_learning_rate}")
    step_size = int(n_epochs * 0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print(f"step_size = {step_size}")

    c, kvp = count_parameters(model)
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

    c, kvp = count_parameters(model)
    for k, v in kvp.items():
        print(f"{k} = {v.item()}")