import os
import sys

import torch
from deepinv.models import GSDRUNet
from deepinv.models.GSPnP import GSPnP

if "/.fork" in sys.prefix:
    sys.path.append('/projects/UDIP/nils_src/deepinv')

import deepinv as dinv
from torch import nn
from torchvision import transforms

from multilevel.approx_nn import Student
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
        else:  # cosine
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


class GSPnPStudent(dinv.models.GSPnP):
    def __init__(self, denoiser):
        super().__init__(denoiser=denoiser)
        self.internal_out = 0

    def forward(self, x, sigma):
        res = super().forward(x, sigma)
        self.internal_out = self.student_grad.denoiser.internal_out
        return res


class GSDrunetTeacher:
    def __init__(self, device):
        self.gs = GSDRUNet(pretrained="download", device=device)

    def forward_unet(self, x0):
        denoiser = self.gs.student_grad.denoiser

        x1 = denoiser.m_head(x0)
        x2 = denoiser.m_down1(x1)
        x3 = denoiser.m_down2(x2)
        x4 = denoiser.m_down3(x3)
        x = denoiser.m_body(x4)
        x = denoiser.m_up3(x + x4)
        x = denoiser.m_up2(x + x3)
        x = denoiser.m_up1(x + x2) # (64, H, W)

        return x + x1


if __name__ == '__main__':
    device = 'cuda:0'
    teacher = DrunetTeacher().to(device)
    #student = Student(layers=6).to(device)
    student = Student(layers=10, nc=32).to(device)
    student.train()

    # todo : test GS teacher
    teacher2 = GSDrunetTeacher(device=device)
    student2 = GSPnPStudent(denoiser=student)

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma=.1))

    mode = 'cs'
    #mode = 'mse'

    # todo : exp. high weight KD
    # todo : exp. no KD
    losses = [KDLoss(teacher, mode=mode, weight=1), dinv.loss.SupLoss()]

    #img_size = 64
    img_size = 128
    #path = '/projects/UDIP/deepinv/datasets/LIDC/divfree/inpainting/'
    path = os.path.join(dataset_path(), 'DIV2K')

    save_path = checkpoint_path()
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop((img_size, img_size))])


    sigma_generator = dinv.physics.generator.SigmaGenerator(sigma_min=0.01, sigma_max=0.2, device=device)
    dataset = dinv.datasets.DIV2K(path, transform=transform, mode='train', download=True)
    eval_dataset = dinv.datasets.DIV2K(path, transform=transform, mode='val', download=True)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=False)

    wandb_setup = {
        'name': f'student_teacher_{mode}_c32_ic2_10L',
        'project': 'student_teacher'}

    trainer = CustomTrainer(wandb_vis=True, wandb_setup=wandb_setup, losses=losses, model=student, ckp_interval=5,
                         physics=physics, verbose_individual_losses=True, ckpt_pretrained=None,
                         save_path=save_path, online_measurements=True, physics_generator=sigma_generator,
                         scheduler=scheduler, optimizer=optimizer, train_dataloader=train_dataloader,
                         device=device, eval_dataloader=eval_dataloader, eval_interval=5, epochs=600)


    trainer.train()

