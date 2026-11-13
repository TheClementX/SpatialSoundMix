"""
losses.py 

the necessary losses for training BSS models
"""

from .stft_modules import * 

class MAELoss(nn.Module): 
    def __init__(
        self, 
        hop_length: int = 147, 
        resolutions: list[int] = [4096, 2048, 1024, 512, 256], 
        channels: int = 1, 
        window: str = "hann" 
    ) -> None: 
        super().__init__()

        self.transforms = nn.ModuleList([
            STFTModule(
                n_fft, 
                hop_length, 
                n_fft, 
                channels=channels, 
                window=window
            ) for n_fft in resolutions
        ])

    """
    @brief: 
        computes the average MAE loss over the batch. 
    @params: 
        predicted_audio (B, channels, L), target_audio (B, channels, L)
    @returns
    """
    def forward(
        self, 
        predicted_audio: torch.Tensor, 
        target_audio: torch.Tensor
    ) -> torch.Tensor: 
        #combine batch and channel Bc = B * channels
        L = predicted_audio.shape[-1]
        l2_pred_aud = predicted_audio.reshape(-1, L)
        l2_tar_aud = target_audio.reshape(-1, L)

        #(B*chan)
        wav_dif = l2_tar_aud - l2_pred_aud
        l2_wavform = torch.abs(wav_dif).mean(dim=1)

        spec_loss = []
        for transform in self.transforms: 
            #(B, channels, F, L)
            tar = torch.view_as_complex(transform(target_audio).to(torch.float32))
            pred = torch.view_as_complex(transform(predicted_audio).to(torch.float32))

            #(B, channels, F, L)
            spec_dif = tar - pred
            l2_spec = torch.abs(spec_dif).mean(dim=(2,3))
            spec_loss.append(l2_spec)
        
        #(S, B, channels)
        spec_loss = torch.stack(spec_loss, dim=0)
        l2_spectrogram = spec_loss.sum(dim=0).view(-1)

        loss = l2_wavform + l2_spectrogram
        return loss.mean()


class SA_SDR(nn.Module): 
    def __init__(self): 
        super().__init__(eps: float = 1e-8)
        self.eps = eps

    """
    @params: 
        sources: (B, num_speakers, len)
        targets: (B, num_speakers, len)
    """
    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor: 
        #(B,)
        target_l2_squared = torch.sum(targets ** 2, dim=(1, 2))
        #(B,)
        difference_l2_squared = torch.sum((targets - predictions)**2, dim=(1,2))
        #(B,)
        per_batch_sa_sdr = -10 * (
            torch.log10(target_l2_squared + self.eps)
            - torch.log10(difference_l2_squared + self.eps)
        )
        #(1,)
        batch_sa_sdr = torch.mean(per_batch_sa_sdr)

        return batch_sa_sdr


