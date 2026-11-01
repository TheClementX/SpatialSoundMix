"""
stft_modules.py 

modules for ease of use in performing STFT operations in speech separation
tasks
"""

import torch
import torch.nn as nn 

"""
STFTModule

@brief: 
    a module to compute the STFT over multi-channel audio. 
    This module does not handle the necessary reshaping for
    passing to MLPs as this must be done after band splitting. 
"""
class STFTModule(nn.Module): 
    def __init__(
        self, 
        n_fft: int, 
        hop_length: int, 
        win_length: int, 
        channels: int = 1,
        window: str = "hann_window"
    ) -> None: 
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.channels = channels
        
        if window == "hann_window":
            window_fn = torch.hann_window(self.win_length)
        else:
            raise ValueError("unsupported window function")
        self.register_buffer("window_fn", window_fn, persistent=False)

    """
    @brief: 
        this operation takes a batch of audio with channels and computes
        the stft for each channel make a batch of channeled spectrograms
        for each input audio. 
    @params: 
        x: (B, channels, L)
    @return: 
        a real tensor where out[..., 0] is the real part and
        out[..., 1] is the imaginary part. (B, channels, F, L_fft, 2). 
        notice that F = n_fft/2 + 1 so the effective spectrogram 
        size is half n_fft. 
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        B, C, L = x.shape
        x = x.contiguous()

        x = x.view(-1, L) #flatten into batches
        complex_fft = torch.stft(
            x, 
            self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window_fn.to(x.device), 
            center=True,
            return_complex=True
        )

        _, F, L_fft = complex_fft.shape
        rebatched = complex_fft.view(B, C, F, L_fft)
        return torch.view_as_real(rebatched)

"""
iSTFTModule

@brief: 
    a module to compute the iSTFT over multi-channel spectrograms.
    This module is meant to be an inverse of the STFTModule. 
"""
class iSTFTModule(nn.Module): 
    def __init__(
        self, 
        n_fft: int, 
        hop_length: int, 
        win_length: int, 
        channels: int = 1,
        window: str = "hann_window"
    ) -> None: 
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.channels = channels
        
        if window == "hann_window":
            window_fn = torch.hann_window(self.win_length)
        else:
            raise ValueError("unsupported window function")
        self.register_buffer("window_fn", window_fn, persistent=False)

    """
    @params: 
        x: (B, channels, F, L_fft, 2) a complex valued ctensor where the last
        dimension is an imaginary number
    @return: 
        a batch of raw audio (B, channels, L)
    """
    def forward(self, x: torch.Tensor, length: int) -> torch.Tensor: 
        B, chan, F, L_fft = x.shape
        x = x.contiguous()
        c_spectrogram = torch.view_as_complex(x)
        c_spectrogram = x.view(-1, F, L_fft)
        audio = torch.istft(
            c_spectrogram, 
            self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window_fn.to(x.device), 
            center=True,
            length=length
        )

        rebatched = audio.view(B, chan, -1)
        return rebatched