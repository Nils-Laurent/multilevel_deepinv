import os
import deepinv
from deepinv.optim import L2

from multilevel.approx_nn import Student
from multilevel.info_transfer import SincFilter
from utils.paths import checkpoint_path

from deepinv.models import DRUNet, GSDRUNet, DnCNN
from multilevel_utils.complex_denoiser import to_complex_denoiser


state_file_v3 = os.path.join(checkpoint_path(), 'student_v3_cs_c32_ic2_10L_525.pth.tar')
state_file_v4 = os.path.join(checkpoint_path(), 'student_v4_cs_c32_ic2_10L_weight2_599.pth.tar')
state_file_1channel = os.path.join(checkpoint_path(), '24-09-20-12:41:35/ckp_599.pth.tar')


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ConfParam(metaclass=Singleton):
    win = None
    levels = None
    iters_fine = None
    iml_max_iter = None
    coarse_iters_ini = None
    use_complex_denoiser = None
    data_fidelity = None
    data_fidelity_lipschitz = None
    denoiser_in_channels = None
    s1coherent_algorithm = None
    s1coherent_init = None
    iter_coarse_pnp_map = None
    iter_coarse_pnp_pgd = None
    iter_coarse_tv = None
    iter_coarse_red = None
    stepsize_multiplier_pnp = None

    def reset(self):
        self.win = SincFilter()
        self.levels = 4
        self.iters_fine = 200
        self.iml_max_iter = 2
        self.coarse_iters_ini = 5
        self.use_complex_denoiser = False
        self.data_fidelity = L2
        self.data_fidelity_lipschitz = 1.0  # data-fidelity Lipschitz cst
        self.denoiser_in_channels = 3
        self.s1coherent_algorithm = True
        self.s1coherent_init = False
        self.iter_coarse_pnp_map = 3
        self.iter_coarse_pnp_pgd = 3
        self.iter_coarse_tv = 3
        self.iter_coarse_red = 3
        self.stepsize_multiplier_pnp = 1.0

    def get_drunet(self, device):
        net = DRUNet(in_channels=self.denoiser_in_channels, out_channels=self.denoiser_in_channels, pretrained="download", device=device)
        denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        if self.use_complex_denoiser is True:
            denoiser = to_complex_denoiser(denoiser, mode="separated")
        denoiser.eval()
        return denoiser

    def get_dncnn_nonexp(self, device):
        net = DnCNN(
            in_channels=self.denoiser_in_channels, out_channels=self.denoiser_in_channels, pretrained="download_lipschitz", device=device
        )
        denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        denoiser.eval()
        return denoiser

    def get_student(self, device):
        d = Student(layers=10, nc=32, cnext_ic=2, pretrained=state_file_v3).to(device)
        if ConfParam().use_complex_denoiser is True:
            d = to_complex_denoiser(d, mode="separated")
        d.eval()
        return d

    def get_student1c(self, device):
        d = Student(in_channels=self.denoiser_in_channels,
                    layers=10, nc=32, cnext_ic=2, pretrained=state_file_1channel).to(device)
        if ConfParam().use_complex_denoiser is True:
            d = to_complex_denoiser(d, mode="separated")
        d.eval()
        return d

    def get_gsdrunet(self, device):
        net = GSDRUNet(pretrained="download", device=device)
        denoiser = net
        #denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        if ConfParam().use_complex_denoiser is True:
            denoiser = to_complex_denoiser(denoiser, mode="separated")
        denoiser.eval()
        return denoiser