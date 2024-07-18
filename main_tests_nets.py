import deepinv
from deepinv.models import DRUNet

from distillation import Student
from gen_fig.fig_drunet_time import fig_net_time_depth, fig_net_time_imgsz
from nets_test.time_measures import measure_net_time_imgsize, measure_net_time_depth
from training.utils import target_psnr_drunet
from utils.device import get_new_device


if __name__ == "__main__":
    print("executing main")
    device = get_new_device()
    print(device)
    sigma_generator = deepinv.physics.generator.SigmaGenerator(sigma_min=0.01, sigma_max=0.2, device=device)
    #target_psnr_drunet('DIV2K_valid_HR', sigma_generator, device, batch_size=1, img_size=64)

    student = Student().to(device)
    drunet = DRUNet(pretrained="download", device=device)

    exp_name = measure_net_time_imgsize(drunet, 'drunet', device)
    fig_net_time_imgsz(exp_name)

    #exp_name = measure_net_time_imgsize(student, net_name='student')
    #fig_net_time_imgsz()

    f_depth = lambda n: Student(layers=n).to(device)
    exp_name = measure_net_time_depth(f_depth, 'student', device)
    fig_net_time_depth(exp_name)

    f_depth = lambda n: drunet
    exp_name = measure_net_time_depth(f_depth, 'drunet', device)
    fig_net_time_depth(exp_name)

