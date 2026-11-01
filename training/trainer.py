import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.amp import GradScaler
from tqdm import tqdm
from torchmetrics.functinal.audio import (
    permutation_invariant_training as pit, 
    pit_permutate, 
    scale_invariant_signal_distortion_ratio as si_sdr
)
from models.stft_modules import * 

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

class SpeechSeparationTrainer: 
    def __init__(
        self, 
        model: nn.Module, 
        num_speakers: int, 
        stft_parameters: dict, 
        optimizer: optim.Optimizer, 
        loss_fn: nn.Module, 
        lr_scheduler: optim.lr_scheudler.LRScheduler, 
        scheduler_step_type: str, 
        train_loader: data.DataLoader, 
        val_loader: data.DataLoader, 
        grad_acum_steps: int,
        mixed_precision: bool = True, 
        use_wandb: bool = True, 
        device: str = "cuda"
    ): 
        self.model = model
        self.num_speakers = num_speakers

        #setup stft layers for preparing model input 
        self.stft_parameters = stft_parameters
        #real valued spectrograms
        self.stft = STFTModule(
            stft_parameters['n_fft'], 
            stft_parameters['hop_length'], 
            stft_parameters['win_length'], 
            stft_parameters['channels'],
            stft_parameters['window']
        )
        #complex valued spectrograms
        self.istft = iSTFTModule(
            stft_parameters['n_fft'], 
            stft_parameters['hop_length'], 
            stft_parameters['win_length'], 
            stft_parameters['channels'],
            stft_parameters['window']
        )

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.scheduler_step_type = scheduler_step_type

        self.train_laoder = train_loader
        self.val_loader = val_loader

        if self.grad_acum_steps < 1: 
            raise ValueError("gradient accumulation steps must be at least 1")
        self.grad_acum_steps = grad_acum_steps
        self.grad_acum = self.grad_acum_steps != 1

        self.mixed_precision = mixed_precision 
        self.scaler = torch.amp.GradScaler(device, enabled=True)
        
        self.use_wandb = use_wandb
        self.device = device

    """
    need to reimplement overlap add average with STF windows 
    """
    def multi_speaker_overlap_add_average(self): 
        pass

    def train_epoch(self): 
        self.model.train()
        self.optimizer.zero_grad()
        loss_meter = AverageMeter()
        batch_loss = 0

        progress = tqdm(
            enumerate(self.train_loader), 
            desc="Training Loop", 
            dynamic_ncols=True, 
            leave=False, 
            total=len(self.train_loader)
        )

        current_mem = 0
        peak_mem = 0
        total_norm=None
        for i, example in progress: 
            mix, targets = example
            mix, targets = targets.to(self.device), mix.to(self.device)

            #process with stft 
            stft_mix = self.stft(mix)

            #mixed precision calculate loss
            with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                tf_predictions = self.model(stft_mix)
                t_predictions = self.istft(tf_predictions)
                loss = pit(
                    t_predictions, targets, self.loss_fn, 
                    mode="speaker_wise", eval_func="min"
                )
                if self.grad_acum: 
                    loss /= self.grad_acum_stepsacum_steps

            #set batch loss based on gradient accumulation
            if self.grad_acum: batch_loss += loss.item()
            else: batch_loss = loss.item()

            #backward pass
            self.scaler.scale(loss).backward()

            #gradient descent step 
            if (
                self.grad_acum and (i+1) % self.grad_acum_steps == 0 or 
                self.grad_acum and (i+1) == len(self.dataloader) or
                not self.grad_acum
            ): 
                loss_meter.update(batch_loss)
                #gradient clipping
                self.scaler.unscale_(self.optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                #update step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.batch_scheduler: self.scheduler.step()
                batch_loss = 0

            #track memory usage
            if self.device == "cuda":
                current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

            progress.set_postfix(
                loss=f"{loss_meter.val:.2f}", 
                avg_loss=f"{loss_meter.avg:.2f}", 
                lr=f"{self.optimizer.param_groups[0]['lr']:.6f}", 
                cur_mem=f"{current_mem:.2f}GB", 
                max_mem=f"{peak_mem:.2f}GB", 
                total_norm=f"{total_norm}"
            )


        if not self.batch_scheduler: self.scheduler.step()

        metrics = {
            "train_avg_loss": loss_meter.avg, 
            "train_peak_mem": peak_mem, 
            "lr": self.optimizer.param_groups[0]['lr'], 
            "total_norm": total_norm
        }

        return metrics

    def validate_epoch(self): 
        pass

    def train(self): 
        pass 

    def infer_separations(
        self, 
        inference_instances: data.DataLoader
    ) -> dict[str, torch.Tensor]: 
        pass 
