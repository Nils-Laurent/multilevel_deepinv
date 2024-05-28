import deepinv
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.paths import dataset_path


class ParallelGaussianNoise(torch.nn.Module):
    def __init__(self, sigma_min=0.0, sigma_max=0.2, sigma=None, x_shape=None, x_device=None):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        if sigma is None:
            sigma = (
                    torch.rand((x_shape[0], 1) + (1,) * (len(x_shape) - 2))
                    * (self.sigma_max - self.sigma_min)
                    + self.sigma_min
            )
            self.sigma = sigma.to(x_device)

    def forward(self, x):
        noise = torch.randn_like(x) * self.sigma[int(x.get_device()*x.shape[0]):int((x.get_device()+1)*x.shape[0])].to(x.device)
        return x + noise


def create_physics(device, gpu_num, batch_shape=None):
    """
    Creates the physics of the problem to be solved.

    :param float sigma_max: max noise power.
    :param str device: gpu/cpu device.
    :param int gpu_num: number of GPUs.
    :param tuple batch_shape: shape of the batch (necessary for initialization of the noise model when using multiple GPUs).
    :return: physics operator
    """

    physics = deepinv.physics.DecomposablePhysics()
    physics.noise_model = deepinv.physics.GaussianNoise()

    #if gpu_num > 1:

    #    x_init_physics = torch.randn(batch_shape).to(device)

    #    physics = deepinv.physics.DecomposablePhysics()
    #    physics.noise_model = ParallelGaussianNoise(sigma_max=sigma_max, x_shape=x_init_physics.shape,
    #                                                x_device=x_init_physics.device)
    #    physics = physics.to(device)
    #    physics = torch.nn.DataParallel(physics, device_ids=list(range(gpu_num)))

    return physics


def get_transforms(train_patch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(train_patch_size, pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    in_channels, out_channels = 3, 3
    val_transform = transforms.ToTensor()

    return train_transform, val_transform, in_channels, out_channels

def load_data(train_data_path, test_data_path, train_transform, val_transform, train_batch_size, num_workers):
    dataset_train = datasets.ImageFolder(root=train_data_path, transform=train_transform)
    dataset_val = datasets.ImageFolder(root=test_data_path, transform=val_transform)

    pin_memory = True if torch.cuda.is_available() else False

    train_dataloader = DataLoader(
        dataset_train, batch_size=train_batch_size, num_workers=num_workers, shuffle=True, pin_memory=pin_memory
    )
    val_dataloader = DataLoader(
        dataset_val, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=pin_memory
    )  # batch_size = 1 for testing because of different image sizes

    return train_dataloader, val_dataloader