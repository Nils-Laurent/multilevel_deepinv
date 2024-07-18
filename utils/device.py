import deepinv
import torch


def get_new_device():
    device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    return device
