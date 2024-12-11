import os
import deepinv
from deepinv.optim import L2

from multilevel.approx_nn import Student
from multilevel.info_transfer import SincFilter
from utils.paths import checkpoint_path

from deepinv.models import DRUNet, GSDRUNet, DnCNN, UNet, SCUNet
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
    inpainting_ratio = None
    use_equivariance = None

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
        self.inpainting_ratio = 0.5
        self.use_equivariance = True

    def get_drunet(self, device):
        # DRUNet : dilated residual UNet
        net = DRUNet(in_channels=self.denoiser_in_channels, out_channels=self.denoiser_in_channels, pretrained="download", device=device)
        denoiser = net
        if self.use_equivariance:
            denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        if self.use_complex_denoiser is True:
            denoiser = to_complex_denoiser(denoiser, mode="separated")
        denoiser.eval()
        return denoiser

    def get_scunet(self, device):
        net = SCUNet(
            in_nc=self.denoiser_in_channels, device=device, pretrained="download"
        )
        net.to(device)
        denoiser = net
        if self.use_equivariance:
            denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        denoiser.eval()
        return denoiser

    def get_dncnn(self, device):
        # DnCNN : denoising convolutional neural network
        net = DnCNN(
            in_channels=self.denoiser_in_channels, out_channels=self.denoiser_in_channels,
            pretrained="download", device=device
        )
        denoiser = net
        if self.use_equivariance:
            denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        denoiser.eval()
        return denoiser

    def get_dncnn_nonexp(self, device):
        net = DnCNN(
            in_channels=self.denoiser_in_channels, out_channels=self.denoiser_in_channels,
            pretrained="download_lipschitz", device=device
        )
        denoiser = net
        if self.use_equivariance:
            denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        denoiser.eval()
        return denoiser

    def get_gsdrunet(self, device):
        net = GSDRUNet(pretrained="download", device=device)
        denoiser = net
        if self.use_equivariance:
            denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        if ConfParam().use_complex_denoiser is True:
            denoiser = to_complex_denoiser(denoiser, mode="separated")
        denoiser.eval()
        return denoiser

    def get_student(self, device):
        net = Student(layers=10, nc=32, cnext_ic=2, pretrained=state_file_v3).to(device)
        denoiser = net
        if self.use_equivariance:
            denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        if ConfParam().use_complex_denoiser is True:
            denoiser = to_complex_denoiser(denoiser, mode="separated")
        denoiser.eval()
        return denoiser

    def get_student1c(self, device):
        net = Student(in_channels=self.denoiser_in_channels,
                    layers=10, nc=32, cnext_ic=2, pretrained=state_file_1channel).to(device)
        denoiser = net
        if self.use_equivariance:
            denoiser = deepinv.models.EquivariantDenoiser(net, random=True)
        if ConfParam().use_complex_denoiser is True:
            denoiser = to_complex_denoiser(denoiser, mode="separated")
        denoiser.eval()
        return denoiser

    def default_param(self):
        params_algo = {
            'cit': self.win,
            'scale_coherent_grad': self.s1coherent_algorithm,
            'scale_coherent_grad_init': self.s1coherent_init,
        }
        return params_algo

class FixedParams(metaclass=Singleton):
    g_param = None
    stepsize_coeff = None

    def reset(self):
        self.g_param = None
        self.stepsize_coeff = None

    def get_g_param(self):
        print(f"USE FIXED PARAMS g_param {self.g_param}")
        return self.g_param

    def get_stepsize_coeff(self):
        print(f"USE FIXED PARAMS stepsize_coeff {self.stepsize_coeff}")
        return self.stepsize_coeff

    def get_str(self):
        str = ""
        if self.g_param is not None:
            str += f"gp{self.g_param}"
        if self.stepsize_coeff is not None:
            str += f"sz{self.stepsize_coeff}"

        if str != "":
            str = "FPAR_" + str + "_"

        return str