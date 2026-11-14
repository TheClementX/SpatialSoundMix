"""
losses.py 

the necessary losses for training BSS models
"""

from .stft_modules import * 
import itertools
from scipy.optimize import linear_sum_assignment

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

"""
@brief: realigns predictions to match targets based on similarity 
score with hungarian algorithm. Then computes SA_SDR

@note: propably prone to breif errors
"""
class hungarian_SA_SDR(nn.Module): 
    def __init__(self, eps: float = 1e-8): 
        super().__init__()
        self.eps = eps

    """
    @params: 
        sources: (B, num_speakers, len)
        targets: (B, num_speakers, len)
    """
    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor: 
        B, C, T = predictions.shape

        #rearrange predictions according to similarity scores
        u_predictions = predictions.unsqueeze(1) #(B, 1, C, T)
        u_targets = targets.unsqueeze(2) #(B, C, 1, T)
        cost_matrix = torch.abs(u_targets - u_predictions).mean(dim=-1)
        cost_matrix_np = cost_matrix.detach().cpu().numpy()

        aligned_preds_per_batch = []

        #permute each batch predition to correct order
        for b in range(B): 
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np[b])
            aligned_preds_per_batch.append(predictions[b, col_ind, :])

        aligned_preds = torch.stack(aligned_preds_per_batch, dim=0)


        #(B,)
        target_l2_squared = torch.sum(targets ** 2, dim=(1, 2))
        #(B,)
        difference_l2_squared = torch.sum((targets - aligned_preds)**2, dim=(1,2))
        #(B,)
        per_batch_sa_sdr = 10 * (
            torch.log10(target_l2_squared + self.eps)
            - torch.log10(difference_l2_squared + self.eps)
        )
        #(1,)
        batch_sa_sdr = torch.mean(per_batch_sa_sdr)

        return batch_sa_sdr

"""
@brief: calculates minimum SA_SDR over all possible pairs of audio
make sure to negate if training
"""
class factorial_SA_SDR(nn.Module): 
    def __init__(self, num_speakers: int, eps: float = 1e-8): 
        super().__init__()
        self.eps = eps

        self.permutations = list(itertools.permutations(range(num_speakers)))

    """
    @params: 
        sources: (B, num_speakers, len)
        targets: (B, num_speakers, len)
    """
    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor: 
        #compute loss over all permutations and take minimum

        B = predictions.shape[0]

        #(B,)
        target_l2_squared = torch.sum(targets ** 2, dim=(1, 2))
        target_log = torch.log10(target_l2_squared + self.eps)

        losses = torch.zeros(
            B, len(self.permutations), device=predictions.device
        )

        #calculate loss per permutation
        for i, p in enumerate(self.permutations): 
            #permute predictions 
            p_preds = predictions[:, p, :]
            #(B,)
            diff_l2_squared = torch.sum((targets - p_preds) ** 2, dim = (1, 2))
            loss = 10 * (target_log - torch.log10(diff_l2_squared + self.eps))
            losses[:, i] = loss

        min_loss, min_idx = torch.min(losses, dim=1)
        return min_loss.mean()