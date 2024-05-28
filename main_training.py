import sys

if "/.fork" in sys.prefix:
    sys.path.append('/projects/UDIP/nils_src/deepinv')

import torch

import deepinv

from training.train_drunet import train_drunet

if __name__ == "__main__":
    #gen_scale_dataset(dataset_name='celeba_hq', scale_count=1)
    #gen_scale_dataset(dataset_name='celeba_hq', scale_count=2)
    #gen_scale_dataset(dataset_name='celeba_hq', scale_count=3)

    #device = deepinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    train_drunet('celeba_hq_coarse3')