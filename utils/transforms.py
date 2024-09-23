import torch

# Centered, ortho ifft
def ifft(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifftn(x, dim=(-2, -1), norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x

# Centered, ortho fft
def fft(x):
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    return x

class IFFTTransform:
    def __call__(self, tensor):
        return ifft(tensor)


def transform_ifft():
    return IFFTTransform()

class SeparateTransform:
    def __call__(self, tensor):
        return torch.cat([tensor.real, tensor.imag], dim=0)

def transform_separate():
    return SeparateTransform()

class CatZeroChannel:
    def __call__(self, tensor):
        c, w, h = tensor.shape
        zeros = torch.zeros((1, w, h), dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, zeros), dim=0)