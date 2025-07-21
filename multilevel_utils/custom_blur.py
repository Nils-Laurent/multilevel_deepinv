from deepinv.physics import Blur

import torch

class CBlur(Blur):
    def to_coarse(self, ds):
        r"""
        Applies the downsampling operator on the blur filter, defining the coarse blur filter.

        :return: deepinv.physics.Blur: coarse blur physics.
        """
        #ds = self.get_downsampling_operator()
        filt = coarse_blur_filter(self.filter, ds.filter)
        return CBlur(filter=filt, padding=self.padding, device=self.filter.device)


def coarse_blur_filter(fine_filter, downsampling_filter):
    in_filt = fine_filter

    # ensure filter is at least 3x3
    if in_filt.shape[-2] <= 3:
        in_filt = torch.nn.functional.pad(in_filt, (0, 0, 1, 1))
    if in_filt.shape[-1] <= 3:
        in_filt = torch.nn.functional.pad(in_filt, (1, 1, 0, 0))

    # left, right, top, bottom padding to perform valid convolution
    df_shape = downsampling_filter.shape
    pad_size = (df_shape[-2] // 2,) * 2 + (df_shape[-1] // 2,) * 2
    pf = torch.nn.functional.pad(in_filt, pad_size)

    # downsample the blur filter
    df_groups = downsampling_filter.repeat([pf.shape[1]] + [1] * (len(pf.shape) - 1))
    coarse_filter = torch.nn.functional.conv2d(
        pf, df_groups, groups=pf.shape[1], padding="valid"
    )
    coarse_filter = coarse_filter[:, :, ::2, ::2]
    coarse_filter = coarse_filter / torch.sum(coarse_filter) * torch.sum(in_filt)

    return coarse_filter