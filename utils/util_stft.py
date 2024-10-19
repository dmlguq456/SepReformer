import torch
import numpy as np

from math import ceil
from dataclasses import dataclass, field, fields
from loguru import logger
from utils.decorators import *


class STFTBase(torch.nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it 
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """
    def __init__(self,
                 frame_length,
                 frame_shift,
                 device="cuda",
                 normalize=False):
        super(STFTBase, self).__init__()
        K = self.init_kernel(frame_length, frame_shift)
        self.K = torch.nn.Parameter(K, requires_grad=False).to(device)
        self.stride = frame_shift
        self.N = frame_length
        self.normalize = normalize
        self.num_bins = self.K.shape[0] // 2

    def extra_repr(self):
        return (f"stride={self.stride}, " +
                f"kernel_size={self.K.shape[0]}x{self.K.shape[2]}, " +
                f"normalize={self.normalize}")
        
    def init_kernel(self, frame_len, frame_hop):
        # FFT points
        N = frame_len
        # window
        W = torch.hann_window(frame_len)
        if N//4 == frame_hop:
            W = (2/3)**0.5*W
            S =  0.5**(3/2) * N / frame_hop**0.5
            # S =  0.5**(3/2) * N**0.5 / frame_hop**0.5
        elif N//2 == frame_hop:
            W = W**0.5
            S = 0.5 * (N * N / frame_hop)**0.5

        K = torch.fft.rfft(torch.eye(N) / S, dim=1)[:frame_len]
        K = torch.stack((torch.real(K), torch.imag(K)),dim=2)
        # 2 x N/2+1 x F
        K = torch.transpose(K, 0, 2) * W
        # N+2 x 1 x F
        K = torch.reshape(K, (N + 2, 1, frame_len))

        return K


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x, cplx=False):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        args
            x: input signal, N x C x S or N x S
        return
            m: magnitude, N x C x F x T or N x F x T
            p: phase, N x C x F x T or N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d}D signal".format(
                    self.__name__, x.dim()))
        # if N x S, reshape N x 1 x S
        if self.normalize:
            x = x
        else:
            x = x * self.N**0.5

        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
            # N x 2F x T
            c = torch.nn.functional.conv1d(x, self.K.to(x.device), stride=self.stride, padding=0)
            # N x F x T
            r, i = torch.chunk(c, 2, dim=1)
        else:
            N, C, S = x.shape
            x = x.reshape(N * C, 1, S)
            # NC x 2F x T
            c = torch.nn.functional.conv1d(x, self.K.to(x.device), stride=self.stride, padding=0)
            # N x C x 2F x T
            c = c.reshape(N, C, -1, c.shape[-1])
            # N x C x F x T
            r, i = torch.chunk(c, 2, dim=2)
        if cplx:
            return torch.complex(r, i)
        m = (r**2 + i**2)**0.5
        p = torch.atan2(i, r)
        return m, p


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p=None, cplx=False, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        args
            m, p: N x F x T
        return
            s: N x S
        """
        if p != None:
            if p.dim() != m.dim() or m.dim() not in [2, 3]:
                raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                    p.dim()))
        # if F x T, reshape 1 x F x T
        if m.dim() == 2:
            if p != None: p = torch.unsqueeze(p, 0)
            m = torch.unsqueeze(m, 0)
        if cplx:
            # N x 2F x T
            c = torch.cat([m.real, m.imag], dim=1)
        else:
            r = m * torch.cos(p)
            i = m * torch.sin(p)
            # N x 2F x T
            c = torch.cat([r, i], dim=1)
        # N x 2F x T
        if self.normalize:
            c = c
        else:
            c = c / self.N**0.5
        # if self.normalize:
        #     c = c * self.N**0.5
        s = torch.nn.functional.conv_transpose1d(c, self.K.to(c.device), stride=self.stride, padding=0)
        # N x S
        s = s.squeeze(1)
        if squeeze:
            s = torch.squeeze(s)
        return s
    