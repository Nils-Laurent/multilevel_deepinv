import deepinv
from deepinv.models import DRUNet, GSDRUNet
from torch import nn

from gen_fig.fig_drunet_time import fig_net_time_depth, fig_net_time_imgsz
from multilevel.prior import TVPrior
from nets_test.time_measures import measure_net_time_imgsize, measure_net_time_depth
from training.utils import target_psnr_drunet


import torch
from multilevel.approx_nn import Student
import torchprofile
import thop


class TestTVTime(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.tv = TVPrior(def_crit=1e-6, n_it_max=1000)
        self.gamma = gamma

    def forward(self, x, sigma=1.0):
        return self.tv.moreau_grad(x, gamma=self.gamma)


def test_nets_time(device):
    d = Student(layers=10, nc=32, cnext_ic=2).to(device)
    d2 = DRUNet(device=device)
    d3 = GSDRUNet(device=device)

    #par_d = sum(p.numel() for p in d.parameters())
    #par_d2 = sum(p.numel() for p in d2.parameters())
    #par_d3 = sum(p.numel() for p in d3.parameters())

    #print(f"Student par {par_d}")
    #print(f"DRUNet par {par_d2}")
    #print(f"GSDRUNet par {par_d3}")

    sigma = torch.tensor(0.05, dtype=torch.float32).to(device)
    x = torch.randn(1, 3, 512, 512, dtype=torch.float32).to(device)
    with torch.no_grad():
        macs1b = torchprofile.profile_macs(d, args=(x, sigma)) # (Multiply-Accumulate Operations)
        macs2b = torchprofile.profile_macs(d2, args=(x, sigma)) # (Multiply-Accumulate Operations)
        macs1, params = thop.profile(d, inputs=(x, sigma))
        macs2, params = thop.profile(d2, inputs=(x, sigma))
        print(f"Student MACS {macs1}")
        print(f"Student MACS b : {macs1b}")
        print(f"DRUNet MACS {macs2}")
        print(f"DRUNet MACS b : {macs2b}")

        dtv = TestTVTime(gamma=0.2).to(device)
        macstv, params = thop.profile(dtv, inputs=(x, sigma))
        macstvb = torchprofile.profile_macs(dtv, x) # (Multiply-Accumulate Operations)
        print(f"TV MACS {macstv}")
        print(f"TV MACS b : {macstvb}")

        #macs3 = torchprofile.profile_macs(d3, args=(x, sigma)) # (Multiply-Accumulate Operations)
        macs3, params = thop.profile(d3, inputs=(x, sigma))
        print(f"GSDRUNet MACS {macs3}")

def main_func():
    print("executing main")
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(device)
    test_nets_time(device)
    return None
    sigma_generator = deepinv.physics.generator.SigmaGenerator(sigma_min=0.01, sigma_max=0.2, device=device)
    #target_psnr_drunet('DIV2K_valid_HR', sigma_generator, device, batch_size=1, img_size=64)

    gsdrunet = GSDRUNet(pretrained="download", device=device)
    drunet = DRUNet(pretrained="download", device=device)

    #exp_name = measure_net_time_imgsize(drunet, 'drunet', device)
    #fig_net_time_imgsz(exp_name)

    #exp_name = measure_net_time_imgsize(student, net_name='student')
    #fig_net_time_imgsz()

    f_depth = lambda n: TestTVTime(gamma=0.2).to(device)
    exp_name = measure_net_time_depth(f_depth, 'TVnet', device)
    fig_net_time_depth(exp_name)

    f_depth = lambda n: Student(layers=n, nc=32, cnext_ic=2).to(device)
    exp_name = measure_net_time_depth(f_depth, 'student32', device)
    fig_net_time_depth(exp_name)

    f_depth = lambda n: drunet
    exp_name = measure_net_time_depth(f_depth, 'drunet', device)
    fig_net_time_depth(exp_name)

    f_depth = lambda n: gsdrunet
    exp_name = measure_net_time_depth(f_depth, 'gsdrunet', device)
    fig_net_time_depth(exp_name)


if __name__ == "__main__":
    main_func()