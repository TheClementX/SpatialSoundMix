import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.amp import GradScaler

class SpeechSeparationTrainer: 
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: optim.Optimizer, 
        loss_fn: nn.Module, 
        lr_scheduler: optim.lr_scheudler.LRScheduler, 
        train_loader: data.DataLoader, 
        val_loader: data.DataLoader, 
        grad_acum_steps: int,
        mixed_precision: bool, 
    ): 
        pass

    def train_step(self): 
        pass

    def inference_step(self): 
        pass

    def train_epoch(self): 
        pass

    def validate_epoch(self): 
        pass

    def train(self): 
        pass 

    def infer_separations(
        self, 
        inference_instances: data.DataLoader
    ) -> dict[str, torch.Tensor]: 
        pass 
