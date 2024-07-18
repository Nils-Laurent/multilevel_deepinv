import torch
import deepinv as dinv
from torch import nn
from torch.nn import Conv2d
from torchvision import transforms

from utils.paths import get_out_dir
from utils.paths import checkpoint_path, dataset_path


class CustomTrainer(dinv.Trainer):
    def __init__(self, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)

    def model_inference(self, y, physics):
        y = y.to(self.device)
        x_net = self.model(y, physics.noise_model.sigma, update_parameters=True)
        return x_net

    def get_samples_online(self, iterators, g):
        data = next(
            iterators[g]
        )  # In this case the dataloader outputs also a class label

        if type(data) is tuple or type(data) is list:
            x = data[0]
        else:
            x = data

        x = x.to(self.device)
        physics = self.physics[g]

        if self.physics_generator is not None:
            params = self.physics_generator[g].step(x.size(0))
            #sigma = params['sigma'].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            sigma = params['sigma']
            y = physics(x, sigma=sigma)
        else:
            y = physics(x)

        return x, y, physics

class KDLoss(dinv.loss.Loss):
    def __init__(self, teacher, mode='mse', weight=1):
        super(KDLoss, self).__init__()
        self.teacher = teacher
        self.mode = mode
        if mode == 'mse':
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CosineSimilarity()
        self.weight = weight

    def forward(self, y, model, physics, **kwargs):
        features_student = model.internal_out
        batch = y.size(0)
        with torch.no_grad():
            features_teacher = self.teacher(y, physics.noise_model.sigma)

        if self.mode == 'mse':
            return self.weight*self.loss(features_student, features_teacher)
        else: # cosine
            return -self.weight*self.loss(features_student.reshape(batch, -1), features_teacher.reshape(batch, -1))

    def get_samples_online(self, iterators, g):
        r"""
        Get the samples for the online measurements.

        In this setting, a new sample is generated at each iteration by calling the physics operator.
        This function returns a dictionary containing necessary data for the model inference. It needs to contain
        the measurement, the ground truth, and the current physics operator, but can also contain additional data.

        :param list iterators: List of dataloader iterators.
        :param int g: Current dataloader index.
        :returns: a tuple containing at least: the ground truth, the measurement, and the current physics operator.
        """
        data = next(
            iterators[g]
        )  # In this case the dataloader outputs also a class label

        if type(data) is tuple or type(data) is list:
            x = data[0]
        else:
            x = data

        x = x.to(self.device)
        physics = self.physics[g]

        if self.physics_generator is not None:
            params = self.physics_generator[g].step(x.size(0))
            y = physics(x, **params)
        else:
            y = physics(x)

        return x, y, physics

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, bias=False, ksize=7,
                 padding_mode='circular', batch_norm=False):
        super().__init__()

        ic = 2

        self.conv1 = Conv2d(in_channels, in_channels, kernel_size=ksize, groups=in_channels,
                               stride=1, padding=ksize // 2, bias=bias, padding_mode=padding_mode)
        if batch_norm:
            self.BatchNorm = nn.BatchNorm2d(in_channels)
        else:
            self.BatchNorm = nn.Identity()

        self.conv2 = Conv2d(in_channels, ic*in_channels, kernel_size=1, stride=1, padding=0, bias=bias, padding_mode=padding_mode)

        self.nonlin = nn.GELU()
        self.conv3 = Conv2d(ic*in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.BatchNorm(out)
        out = self.nonlin(out)
        out = self.conv3(out)
        return out + x


class Student(torch.nn.Module):
    def __init__(
            self,
            layers=5,
            pretrained=None,
    ):
        super(Student, self).__init__()

        self.convin = Conv2d(4, 64, kernel_size=1)
        self.convout = Conv2d(64, 3, kernel_size=1)

        self.net = torch.nn.Sequential()
        self.internal_out = 0

        for i in range(layers):
            self.net.add_module(f'block_{i}', ConvNextBlock(in_channels=64))

        if pretrained is not None:
            ckpt_drunet = torch.load(
                pretrained, map_location=lambda storage, loc: storage
            )
            self.load_state_dict(ckpt_drunet['state_dict'], strict=True)
            self.eval()

    def internal_forward(self, x, sigma):
        if isinstance(sigma, torch.Tensor):
            if sigma.ndim > 0:
                noise_level_map = sigma.view(x.size(0), 1, 1, 1)
                noise_level_map = noise_level_map.expand(-1, 1, x.size(2), x.size(3))
            else:
                noise_level_map = torch.ones(
                    (x.size(0), 1, x.size(2), x.size(3)), device=x.device
                ) * sigma[None, None, None, None].to(x.device)
        else:
            noise_level_map = (
                torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
                * sigma
            )
        x = torch.cat((x, noise_level_map), 1)
        out = self.convin(x)
        out = self.net(out)
        return out

    def forward(self, x, sigma, update_parameters=False):
        out = self.internal_forward(x, sigma)
        if self.training and update_parameters:
            self.internal_out = out
        return self.convout(out) #+ x


class DrunetTeacher(dinv.models.DRUNet):
    def __init__(self):
        super(DrunetTeacher, self).__init__()

    def forward_unet(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2) # (64, H, W)
        return x + x1



if __name__ == '__main__':
    device = 'cuda:0'
    teacher = DrunetTeacher().to(device)
    student = Student().to(device)
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma=.1))

    mode = 'cs'
    #mode = 'mse'

    losses = [KDLoss(teacher, mode=mode), dinv.loss.SupLoss()]
    #losses = [dinv.loss.SupLoss()]

    img_size = 64
    path = '/projects/UDIP/deepinv/datasets/LIDC/divfree/inpainting/'

    save_path = checkpoint_path()
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop((img_size, img_size))])


    sigma_generator = dinv.physics.generator.SigmaGenerator(sigma_min=0.01, sigma_max=0.2, device=device)
    dataset = dinv.datasets.DIV2K(path, transform=transform, mode='train', download=True)
    eval_dataset = dinv.datasets.DIV2K(path, transform=transform, mode='val', download=True)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=False)

    wandb_setup = {
        'name': f'student_teacher_{mode}',
        'project': 'student_teacher'}

    trainer = CustomTrainer(wandb_vis=True, wandb_setup=wandb_setup, losses=losses, model=student, ckp_interval=5,
                         physics=physics, verbose_individual_losses=True, ckpt_pretrained=None,
                         save_path=save_path, online_measurements=True, physics_generator=sigma_generator,
                         scheduler=scheduler, optimizer=optimizer, train_dataloader=train_dataloader,
                         device=device, eval_dataloader=eval_dataloader, eval_interval=5, epochs=300)


    trainer.train()

