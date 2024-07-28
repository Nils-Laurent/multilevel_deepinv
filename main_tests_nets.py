import deepinv
from deepinv.models import DRUNet, GSDRUNet
from torch import nn

from distillation import Student
from gen_fig.fig_drunet_time import fig_net_time_depth, fig_net_time_imgsz
from multilevel.prior import TVPrior
from nets_test.time_measures import measure_net_time_imgsize, measure_net_time_depth
from training.utils import target_psnr_drunet
from utils.device import get_new_device


class TestTVTime(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.tv = TVPrior(def_crit=1e-6, n_it_max=1000)
        self.gamma = gamma

    def forward(self, x):
        return self.tv.moreau_grad(x, gamma=self.gamma)


if __name__ == "__main__":
    print("executing main")
    device = get_new_device()
    print(device)
    sigma_generator = deepinv.physics.generator.SigmaGenerator(sigma_min=0.01, sigma_max=0.2, device=device)
    #target_psnr_drunet('DIV2K_valid_HR', sigma_generator, device, batch_size=1, img_size=64)

    student = Student().to(device)
    #gsdrunet = GSDRUNet(pretrained="download", device=device)
    #drunet = DRUNet(pretrained="download", device=device)

    #exp_name = measure_net_time_imgsize(drunet, 'drunet', device)
    #fig_net_time_imgsz(exp_name)

    #exp_name = measure_net_time_imgsize(student, net_name='student')
    #fig_net_time_imgsz()

    # todo : measure TV
    f_depth = lambda n: TestTVTime(gamma=0.2).to(device)
    exp_name = measure_net_time_depth(f_depth, 'TVnet', device)
    fig_net_time_depth(exp_name)

    f_depth = lambda n: Student(layers=n, nc=64).to(device)
    exp_name = measure_net_time_depth(f_depth, 'student64', device)
    fig_net_time_depth(exp_name)

    f_depth = lambda n: Student(layers=n, nc=32).to(device)
    exp_name = measure_net_time_depth(f_depth, 'student32', device)
    fig_net_time_depth(exp_name)

    #f_depth = lambda n: drunet
    #exp_name = measure_net_time_depth(f_depth, 'drunet', device)
    #fig_net_time_depth(exp_name)

    #f_depth = lambda n: gsdrunet
    #exp_name = measure_net_time_depth(f_depth, 'gsdrunet', device)
    #fig_net_time_depth(exp_name)

