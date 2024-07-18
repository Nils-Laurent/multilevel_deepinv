import torch
import deepinv
from deepinv.models import DRUNet, ArtifactRemoval
from deepinv.physics import GaussianNoise
from deepinv.physics.generator import SigmaGenerator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from training.drunet_utils import get_transforms
from utils.paths import checkpoint_path, dataset_path


def target_psnr_drunet(dataset_name, noise_pow, device, batch_size=None, img_size=128):
    in_channels = 3
    out_channels = 3
    target_network = DRUNet(
        in_channels=in_channels, out_channels=out_channels, pretrained="download", device=device
    ).to(device)
    model = ArtifactRemoval(backbone_net=target_network, device=device)
    physics = deepinv.physics.DecomposablePhysics(device=device)
    physics.noise_model = GaussianNoise()
    if isinstance(noise_pow, SigmaGenerator):
        generator = noise_pow
    else:
        generator = SigmaGenerator(sigma_max=noise_pow, device=device)

    gpu_num = 1
    num_workers = 4 * gpu_num if torch.cuda.is_available() else 0
    if batch_size is None:
        batch_size = 64*gpu_num

    print(f"Batch size: {batch_size}")

    pin_memory = True if torch.cuda.is_available() else False
    val_transform = transforms.Compose([
        transforms.RandomCrop(img_size, pad_if_needed=True),
        transforms.ToTensor(),
    ])
    test_path = dataset_path() / dataset_name
    dataset_val = datasets.ImageFolder(root=test_path, transform=val_transform)
    val_dataloader = DataLoader(
        dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=pin_memory
    )  # batch_size = 1 if image sizes are different

    deepinv.test(
        model,
        test_dataloader=val_dataloader,
        physics=physics,
        physics_generator=generator,
        online_measurements=True,
        device=device
    )