import deepinv
import torch
from deepinv import train
from deepinv.physics import GaussianNoise

from multilevel.info_transfer import BlackmannHarris
from multilevel.iterator import MultiLevelParams
from tests.utils import physics_from_exp, data_from_user_input
from tests.utils import standard_multilevel_param
from tests.test_alg import RunAlgorithm


def tune_param(data_in, params_exp, device):
    lambda_red = 0.1
    g_param = 0.05
    noise_pow = params_exp["noise_pow"]

    g = GaussianNoise(sigma=noise_pow)
    physics, problem_name = physics_from_exp(params_exp, g, device)
    data = data_from_user_input(data_in, physics, params_exp, problem_name, device)

    lip_d = 160

    iters_fine = 80
    iters_vec = [5, 5, 5, iters_fine]
    if device == "cpu":
        iters_vec = [5, 5, iters_fine]

    levels = len(iters_vec)
    p_multilevel = MultiLevelParams({"iters": iters_vec})

    params_algo = {
        'cit': BlackmannHarris(),
        'level': levels,
        'params_multilevel': p_multilevel,
        'iml_max_iter': 8,
    }

    p_red = standard_multilevel_param(params_algo, lambda_red, step_coeff=0.9, lip_g=lip_d)
    p_red['g_param'] = g_param
    p_red['scale_coherent_grad'] = True

    param_init = {'init_ml_x0': [80] * levels}
    param_train = ['g_param', 'lambda']
    ra = RunAlgorithm(data, physics, params_exp, device=device, param_init=param_init, r_model=True, trainable_params=param_train)
    model = ra.RED_GD(p_red)

    p2 = model.parameters()
    p_test = model.named_parameters()
    model_p_learn = list(filter(lambda kv: kv[0] in param_train, model.named_parameters()))

    sgd_learning_rate = 0.01
    optimizer = torch.optim.SGD(params = model_p_learn, lr=sgd_learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LRScheduler(optimizer)

    tune_losses = [deepinv.loss.SupLoss(metric=deepinv.metric.mse())]

    n_epochs = 10
    for epoch in range(n_epochs):
        model, loss_hist, psnr_hist = train(
            model=model,
            train_dataloader=data,
            scheduler = scheduler,
            epochs=1,
            losses=tune_losses,
            physics=physics,
            optimizer=optimizer,
            device=device,
            verbose=True,
            return_loss=True,
            max_pixel_psnr = 255,
        )

        params = model.named_parameters()
        for name in p_learn:
            param = params[name]
            print(f'name {name}, param {param}')
