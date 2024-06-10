import deepinv
import torch
from deepinv.models import DRUNet, ArtifactRemoval
from deepinv.physics import GaussianNoise
from deepinv.physics.generator import SigmaGenerator

from multilevel.drunet_scale import DRUNetScale
from training.drunet_utils import get_transforms, load_data
from training.drunet_scale_utils import create_physics, ScaleModel
from utils.paths import checkpoint_path


def target_psnr(noise_pow, device):
    in_channels = 3
    out_channels = 3
    target_network = DRUNet(
        in_channels=in_channels, out_channels=out_channels, pretrained="download", train=False, device=device
    ).to(device)
    model = ArtifactRemoval(backbone_net=target_network, device=device)
    physics = deepinv.physics.DecomposablePhysics(device=device)
    physics.noise_model = GaussianNoise()
    generator = SigmaGenerator(sigma_max=noise_pow, device=device)

    gpu_num = 1
    num_workers = 8 * gpu_num if torch.cuda.is_available() else 0
    train_patch_size = 128
    train_batch_size = 64*gpu_num

    train_ = "CBSD10"
    test_ = "CBSD68"

    train_transform, val_transform, in_channels, out_channels = get_transforms(train_patch_size)
    train_dataloader, test_dataloader = load_data(
        train_, test_, train_transform, val_transform, train_batch_size, num_workers
    )

    deepinv.test(
        model,
        test_dataloader=test_dataloader,
        physics=physics,
        physics_generator=generator,
        online_measurements=True,
        device=device
    )


def test_drunet_scale():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(device)

    noise_pow = 0.2
    target_psnr(noise_pow, device)

    gpu_num = 1
    num_workers = 8 * gpu_num if torch.cuda.is_available() else 0
    train_patch_size = 128
    train_batch_size = 64*gpu_num
    epochs = 100*train_batch_size
    learning_rate=1e-4

    train_transform, val_transform, in_channels, out_channels = get_transforms(train_patch_size)

    train_ = "CBSD10"
    test_ = "CBSD68"
    train_dataloader, test_dataloader = load_data(
        train_, test_, train_transform, val_transform, train_batch_size, num_workers
    )

    in_channels = 3
    out_channels = 3
    model_net = DRUNetScale(
        in_channels=in_channels, out_channels=out_channels, pretrained=None, train=True, device=device
    ).to(device)
    model = ScaleModel(model_net)

    if gpu_num > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(gpu_num)))

    batch_shape = (train_batch_size, in_channels, train_patch_size, train_patch_size)
    max_scale = 2
    physics = create_physics(max_scale, noise_pow, device, gpu_num, batch_shape)

    # losses = deepinv.loss.SupLoss(metric=deepinv.metric.mse())
    losses = deepinv.loss.SupLoss(metric=deepinv.metric.l1())  # todo : find out why not mse

    operation = "drunet_scale_denoise"
    op_dir = checkpoint_path() / operation

    # choose optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=int(epochs/8))
    trainer = deepinv.Trainer(
        model,
        device=device,
        save_path=str(op_dir),
        verbose=True,
        wandb_vis=False,
        physics=physics,
        epochs=epochs,
        scheduler=scheduler,
        losses=losses,
        optimizer=optimizer,
        show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        eval_interval=100,
        ckp_interval=100,
        online_measurements=True,
        check_grad=True,
        #ckpt_pretrained=ckpt_resume,
        freq_plot=100,
    )

    model = trainer.train()