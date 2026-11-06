import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.utils.data as data
from torch.amp import GradScaler
from tqdm import tqdm
from torchmetrics.functional.audio import (
    permutation_invariant_training as pit, 
    scale_invariant_signal_distortion_ratio as si_sdr
)
from models.stft_modules import * 
from scipy.optimize import linear_sum_assignment
import numpy as np
import os
from typing import * 
import wandb

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
        lr_scheduler: optim.lr_scheduler.LRScheduler, 
        batch_scheduler: bool, 
        train_loader: data.DataLoader, 
        val_loader: data.DataLoader, 
        inf_loader: data.DataLoader, 
        inf_chunk_size: int, 
        inf_hop_size: int, 
        grad_acum_steps: int,
        wandb_key: str, 
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
            stft_parameters['in_channels'],
            stft_parameters['window']
        )
        #complex valued spectrograms
        self.istft = iSTFTModule(
            stft_parameters['n_fft'], 
            stft_parameters['hop_length'], 
            stft_parameters['win_length'], 
            stft_parameters['out_channels'],
            stft_parameters['window']
        )

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = lr_scheduler
        self.batch_scheduler = batch_scheduler

        self.train_loader = train_loader #load clips for training
        self.val_loader = val_loader #load clips for validation
        self.inf_loader = inf_loader #load full tracks for inference

        print(f"train loader shape/s : {[t.shape for t in next(iter(self.train_loader))]}")
        print(f"val loader shape/s : {[t.shape for t in next(iter(self.val_loader))]}")
        print(f"inf loader shape/s : {[t.shape for t in next(iter(self.inf_loader))]}")

        self.inf_chunk_size = inf_chunk_size
        self.inf_hop_size = inf_hop_size

        self.grad_acum_steps = grad_acum_steps
        if self.grad_acum_steps < 1: 
            raise ValueError("gradient accumulation steps must be at least 1")
        self.grad_acum = self.grad_acum_steps != 1

        self.mixed_precision = mixed_precision 
        self.scaler = GradScaler(device, enabled=True)
        
        self.wandb_key = wandb_key
        self.use_wandb = use_wandb
        self.device = device

    def audio_to_spatial_net_in(self, mix: torch.Tensor) -> torch.Tensor: 
        #(B, C, F, T, 2)
        audio_spec = self.stft(mix) 
        B, C, F, T, _ = audio_spec.shape
        #(B, F, T, C, 2)
        model_in = audio_spec.permute(0, 2, 3, 1, 4).contiguous()
        #(B, F, T, C*2)
        model_in = model_in.reshape((B, F, T, C*2))
        return model_in

    def spatial_net_out_to_audio(self, model_out: torch.Tensor, length: int) -> torch.Tensor: 
        B, F, T, _ = model_out.shape
        model_out = model_out.float()
        #(B, F, T, 2*num_speakers)
        model_out = model_out.reshape((B, F, T, -1, 2))
        #(B, num_speakers, F, T, 2)
        model_out = model_out.permute(0, 3, 1, 2, 4).contiguous()
        #(B, num_speakers, samples)
        pred_audio = self.istft(model_out, length)
        return pred_audio

    """
    need to reimplement overlap add average with STF windows 

    @param: 
        mix_audio: (B, chan, samples)
    @return
        speakers: (B, num_speakers, samples)
    """
    def multi_speaker_overlap_add_average(
        self, 
        mix_audio: torch.Tensor, 
        num_speakers: int, 
        chunk_size: int = 441000, 
        hop_size: int = 220500 
    ) -> torch.Tensor:
        B, channels, original_seq_len = mix_audio.shape
        pad_len = chunk_size - (original_seq_len % hop_size)

        if pad_len > 0: 
            mix_audio = fn.pad(mix_audio, (0, pad_len))
        seq_len = mix_audio.shape[-1]

        overlap_size = chunk_size - hop_size
        #(B, chan, num_chunks, chunk_size)
        audio_chunks = mix_audio.unfold(dimension=-1, size=chunk_size, step=hop_size)
        #(num_chunks, B, chan, chunk_size)
        audio_chunks = audio_chunks.permute(2, 0, 1, 3)

        window = torch.hann_window(chunk_size, device=mix_audio.device).view(1, 1, -1)
        out_buffer = torch.zeros((B, num_speakers, seq_len), device=mix_audio.device)
        norm_buffer = torch.zeros((B, 1, seq_len), device=mix_audio.device)

        for i, chunk in enumerate(audio_chunks): 
            #predict spectrograms must reshape for input to SpatialNet
            istft_length = chunk.shape[-1]
            model_in = self.audio_to_spatial_net_in(chunk)
            pred_spectrograms = self.model(model_in)
            pred_audio = self.spatial_net_out_to_audio(pred_spectrograms, istft_length)

            #match audio with hungarian if not the first iteration
            start_idx = i * hop_size
            end_idx = start_idx + chunk_size
            if i > 0: 
                #get prefix and suffix for similarity scores
                #normalize over last dim = audio dim
                suffix_window = fn.normalize(out_buffer[:, :, start_idx:start_idx+overlap_size], p=2, dim=2)
                prefix_window = fn.normalize(pred_audio[:, :, :overlap_size], p=2, dim=2)

                #calculate inverted similarity scores
                sim_scores = suffix_window @ prefix_window.permute(0, 2, 1)
                sim_scores = -1 * sim_scores 

                sim_scores = sim_scores.detach().cpu().numpy()
                hungarian_aligned_audio = torch.empty_like(pred_audio)
                for b in range(B): 
                    row_ind, col_ind = linear_sum_assignment(sim_scores[b])
                    hungarian_aligned_audio[b] = pred_audio[b, col_ind, :]

                #create permutation
                pred_audio = hungarian_aligned_audio

            #overlap add
            out_buffer[:, :, start_idx:end_idx] += pred_audio * window
            norm_buffer[:, :, start_idx:end_idx] += window.squeeze(0)

        norm_buffer = torch.where(
            norm_buffer > 1e-8, 
            norm_buffer, 
            torch.ones_like(norm_buffer)
        )

        output = out_buffer / norm_buffer
        #final output has padding stripped. 
        return output[..., :original_seq_len]

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
            mix, targets = mix.to(self.device), targets.to(self.device)
            istft_length = mix.shape[-1]

            model_in = self.audio_to_spatial_net_in(mix)
            with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                tf_predictions = self.model(model_in)
                #reshape
                t_predictions = self.spatial_net_out_to_audio(tf_predictions, istft_length)

                loss, _ = pit(
                    t_predictions.unsqueeze(2), 
                    targets.unsqueeze(2), 
                    self.loss_fn, 
                    mode="speaker-wise", eval_func="min"
                )
                loss = loss.mean()

                if self.grad_acum: 
                    loss /= self.grad_acum_steps

            #set batch loss based on gradient accumulation
            if self.grad_acum: batch_loss += loss.item()
            else: batch_loss = loss.item()

            #backward pass
            self.scaler.scale(loss).backward()

            #gradient descent step 
            if (
                self.grad_acum and (i+1) % self.grad_acum_steps == 0 or 
                self.grad_acum and (i+1) == len(self.train_loader) or
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

    """
    @brief: only compute validation loss
    """
    def validate_epoch(self): 
        self.model.eval()
        loss_meter = AverageMeter()

        progress = tqdm(
            self.val_loader, 
            desc="Validation Loop", 
            dynamic_ncols=True, 
            leave=False, 
        )

        current_mem = 0
        peak_mem = 0
        for example in progress: 
            mix, targets = example
            mix, targets = mix.to(self.device), targets.to(self.device)
            istft_length = mix.shape[-1]

            model_in = self.audio_to_spatial_net_in(mix)
            with torch.amp.autocast(device_type=self.device):
                with torch.inference_mode(): 
                    tf_predictions = self.model(model_in)
                    t_predictions = self.spatial_net_out_to_audio(tf_predictions, istft_length)

                    loss, _ = pit(
                        t_predictions.unsqueeze(2), 
                        targets.unsqueeze(2), 
                        self.loss_fn, 
                        mode="speaker-wise", eval_func="min"
                )

            batch_loss = loss.mean().item()
            loss_meter.update(batch_loss)

            #track memory usage
            if self.device == "cuda":
                current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

            progress.set_postfix(
                loss=f"{batch_loss:.4f}", 
                avg_loss=f"{loss_meter.avg:.4f}", 
                cur_mem=f"{current_mem:.2f}GB", 
                max_mem=f"{peak_mem:.2f}GB"
            )


        metrics = {
            "val_avg_loss": loss_meter.avg, 
            "val_peak_mem": peak_mem, 
        }

        return metrics

    #used only to validate full separations
    def validate_full_separations(
        self
    ) -> dict[str, torch.Tensor]: 
        self.model.eval()
        mae_meter = AverageMeter()
        si_sdr_meter = AverageMeter()

        progress = tqdm(
            self.inf_loader, 
            desc="Separations Validation Loop", 
            dynamic_ncols=True, 
            leave=False, 
        )

        current_mem = 0
        peak_mem = 0
        for example in progress: 
            #full recording pairs
            mix, targets = example
            mix, targets = mix.to(self.device), targets.to(self.device)

            #mixed precision calculate loss
            with torch.amp.autocast(device_type=self.device):
                with torch.inference_mode(): 
                    #(B, num_speakers, len)
                    prediction = self.multi_speaker_overlap_add_average(
                        mix, self.num_speakers, 
                        chunk_size=self.inf_chunk_size, 
                        hop_size=self.inf_hop_size
                    )

                    mae_loss, _ = pit(
                        prediction.unsqueeze(2), 
                        targets.unsqueeze(2), 
                        self.loss_fn, 
                        mode="speaker-wise", eval_func="min"
                    )

                    si_sdr_loss, _ = pit(
                        prediction, 
                        targets, 
                        si_sdr, 
                        mode="speaker-wise", eval_func="max"
                    )

            batch_mae_loss = mae_loss.mean().item()
            batch_si_sdr_loss = si_sdr_loss.mean().item()
            mae_meter.update(batch_mae_loss)
            si_sdr_meter.update(batch_si_sdr_loss)

            #track memory usage
            if self.device == "cuda":
                current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

            progress.set_postfix(
                mae_loss=f"{batch_mae_loss:.4f}", 
                avg_mae_loss=f"{mae_meter.avg:.4f}", 
                si_sdr=f"{batch_si_sdr_loss:.4f}", 
                avg_si_sdr=f"{si_sdr_meter.avg:.4f}", 
                cur_mem=f"{current_mem:.2f}GB", 
                max_mem=f"{peak_mem:.2f}GB"
            )


        metrics = {
            "avg_sep_mae_loss": mae_meter.avg, 
            "avg_sep_si_sdr": si_sdr_meter.avg
        }

        return metrics

    def save_train_state(
        self, 
        path: str, 
        name: str, 
        metric: dict, epoch: int
    ) -> None: 
        torch.save({
            "model_state_dict" : self.model.state_dict(), 
            "optimizer_state_dict" : self.optimizer.state_dict(), 
            "scheduler_state_dict" : self.scheduler.state_dict(), 
            "metric" : metric, 
            "epoch" : epoch, 
        }, os.path.join(path, name))

    def load_train_state(
        self, path: str, 
    ) -> tuple: 
        state = torch.load(path)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])

        return state["epoch"], state["metric"]

    def save_model_state(
        self, name: str, path: str
    ) -> None:
        torch.save(self.model.state_dict(), os.path.join(path, name))

    def load_model_state(
        self, path: str
    ) -> None:
        state = torch.load(path)
        self.model.load_state_dict(state)

    def train(
        self, 
        epochs: int, 
        config: dict,
        model_save_path: str,  
        run_name: str, 
        resume: bool = False, 
        model_load_path: Optional[str] = None
    ) -> None: 
        #setup wandb
        if self.use_wandb: 
            wandb.login(key=self.wandb_key)
            run = wandb.init(
                name=run_name, 
                entity="glundgre-carnegie-mellon-university", 
                project="SpatialSoundMix",
                config=config
            )
            wandb.watch(self.model, log="gradients", log_freq=20)

        if resume: 
            epoch, _ = self.load_train_state(model_load_path)
        else: 
            epoch = 0

        #main training loop
        best_sdr, best_val_loss = None, None
        for i in range(epoch, epochs): 
            print(f"============== EPOCH : {i} =================")
            metrics_and_data = {}
            
            # train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            # metrics_and_data.update(train_metrics)
            metrics_and_data.update(val_metrics)

            if i % 5 == 0: 
                separation_metrics = self.validate_full_separations()
                metrics_and_data.update(separation_metrics)

                #higher is better
                if not best_sdr or metrics_and_data['avg_sep_si_sdr'] > best_sdr: 
                    self.save_model_state('best_sdr_model.pth', model_save_path) 
                    self.save_train_state(
                        model_save_path, 
                        'best_sdr_train_state.pth',
                        metrics_and_data, i
                    )
                    best_sdr = metrics_and_data['avg_sep_si_sdr']

            #lower is better
            if not best_val_loss or metrics_and_data['val_avg_loss'] < best_val_loss: 
                self.save_model_state('best_val_loss_model.pth', model_save_path) 
                self.save_train_state(
                    model_save_path, 
                    'best_val_loss_train_state.pth',
                    metrics_and_data, i
                )
                best_val_loss = metrics_and_data['val_avg_loss']

            #save checkpoint
            self.save_train_state(
                model_save_path, "train_state.pth", 
                metrics_and_data, i 
            )

            print(metrics_and_data)

            #wandb logging
            if self.use_wandb: run.log(metrics_and_data)

        if self.use_wandb: run.finish()

    def infer_separations(
        self, 
        inference_instances: data.DataLoader
    ) -> dict[str, torch.Tensor]: 
        pass 
