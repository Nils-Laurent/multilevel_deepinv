import deepinv
import torch
from deepinv.models import DRUNet, ArtifactRemoval
from deepinv.physics import GaussianNoise
from deepinv.physics.generator import SigmaGenerator
from torch.utils.data import DataLoader
from torchvision import datasets

from training.drunet_utils import get_transforms
from utils.paths import checkpoint_path, dataset_path


def target_psnr(dataset_name, noise_pow, device):
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

    pin_memory = True if torch.cuda.is_available() else False
    train_transform, val_transform, in_channels, out_channels = get_transforms(train_patch_size)
    test_path = dataset_path() / dataset_name / 'val'
    dataset_val = datasets.ImageFolder(root=test_path, transform=val_transform)
    val_dataloader = DataLoader(
        dataset_val, batch_size=train_batch_size, num_workers=num_workers, shuffle=False, pin_memory=pin_memory
    )  # batch_size = 1 if image sizes are different

    deepinv.test(
        model,
        test_dataloader=val_dataloader,
        physics=physics,
        physics_generator=generator,
        online_measurements=True,
        device=device
    )


def train_drunet(dataset_name):
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(device)

    noise_max_pow = 0.2
    target_psnr(dataset_name, noise_max_pow, device)

    gpu_num = 1
    num_workers = 8 * gpu_num if torch.cuda.is_available() else 0
    train_patch_size = 128
    train_batch_size = 64*gpu_num
    epochs = 100*train_batch_size
    learning_rate=1e-4

    train_transform, val_transform, in_channels, out_channels = get_transforms(train_patch_size)

    train_path = dataset_path() / dataset_name / 'train'
    test_path = dataset_path() / dataset_name / 'val'

    dataset_train = datasets.ImageFolder(root=train_path, transform=train_transform)
    dataset_val = datasets.ImageFolder(root=test_path, transform=val_transform)

    pin_memory = True if torch.cuda.is_available() else False
    train_dataloader = DataLoader(
        dataset_train, batch_size=train_batch_size, num_workers=num_workers, shuffle=True, pin_memory=pin_memory
    )
    val_dataloader = DataLoader(
        dataset_val, batch_size=train_batch_size, num_workers=num_workers, shuffle=False, pin_memory=pin_memory
    )  # batch_size = 1 if image sizes are different

    in_channels = 3
    out_channels = 3
    model_net = DRUNet(
        in_channels=in_channels, out_channels=out_channels, pretrained=None, train=True, device=device
    ).to(device)
    model = ArtifactRemoval(backbone_net=model_net, device=device)

    if gpu_num > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(gpu_num)))

    generator = SigmaGenerator(sigma_max=noise_max_pow, device=device)
    physics = deepinv.physics.DecomposablePhysics(device=device)
    physics.noise_model = GaussianNoise()

    losses = deepinv.loss.SupLoss(metric=deepinv.metric.mse())
    #losses = deepinv.loss.SupLoss(metric=deepinv.metric.l1())  # todo : find out why not mse

    operation = "drunet_denoise_" + dataset_name
    op_dir = checkpoint_path() / operation

    # choose optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=int(epochs/8))

    wandb_setup = {
        # set the wandb project where this run will be logged
        'project': "multilevel_deepinv",
        'name': "level training: " + dataset_name,
        # track hyperparameters and run metadata
        'config': {
            "learning_rate": learning_rate,
            "architecture": 'DRUNet',
            "dataset": dataset_name,
            "epochs": epochs,
        }
    }

    #x = torch.rand((train_dataloader.batch_size, 3, 128, 128)).to(device)
    #params = generator.step(x.size(0))
    #y = physics(x, **params)
    #x_hat = model(y, physics)
    #return

    trainer = deepinv.Trainer(
        model,
        device=device,
        save_path=str(op_dir),
        verbose=True,
        wandb_vis=True,
        wandb_setup=wandb_setup,
        physics=physics,
        physics_generator=generator,
        epochs=epochs,
        scheduler=scheduler,
        losses=losses,
        optimizer=optimizer,
        show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        eval_interval=100,
        ckp_interval=100,
        online_measurements=True,
        check_grad=True,
        #ckpt_pretrained=ckpt_resume,
        freq_plot=100,
    )

    model = trainer.train()