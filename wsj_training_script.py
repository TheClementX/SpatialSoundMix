#local imports
from models.losses import MAELoss, factorial_SA_SDR
from models.spatialnet.SpatialNet import * 
from training.datasets import *
from training.wsj_trainer import WsjSpeechSeparationTrainer
from training.sms_wsj_dataset import * 

#pytorch imports
import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr 

"""
interact command: 
srun --account=bjor-delta-gpu --partition=gpuA100x4-interactive --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=01:00:00 --pty bash
"""

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", 
    #model 
    'num_speakers': 2,  
    'num_mics': 6, 
    'dim_squeeze': 16, 
    'num_layers': 12, 
    'num_freqs': 129, 
    'num_heads': 4, 

    #stft
    "stft_parameters": {
        "n_fft": 256,
        "hop_length":  256 // 2,
        "win_length": 256,
        "in_channels": 6, #mixes are binaural
        "out_channels": 1, #separations are monaural
        "window": "hann"
    }, 

    #datasets
    "wsj_root": "/home/clem3nti/projects/SpatialSoundMix/data/sms_wsj", 
    "audio_time_len": 4.0, 
    "target": "direct_path", 
    "sample_rate": 44100, 
    "d_sample_rate": 8000,

    #training
    'lr': 5e-4, 
    'batch_size_train': 4, 
    'batch_size_val': 2, 
    'batch_size_inf': 1,
    'epochs': 40, 
    'warmup_epochs': None, 
    'grad_acum': True, 
    'grad_acum_steps': 1, 
    'mixed_precision': True, 
    'batch_scheduler': True, 
    'checkpointing': True, 
    'full_val_steps': 10, 

    #lr scheduler
    'pct_start': 0.05, 
    'div_factor': 25, 
    'final_div_factor': 1e3, 

    #dataloaders
    'num_workers': 10, 
    'pin_memory': True,
    'persistent_workers': False, 
    'prefetch_factor': 4,

    #inference
    "inf_chunksize": 64000, 
    "inf_hopsize": 64000 // 4, 

    #wandb
    'wandb_key': "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", 
    'use_wandb': True, 
    'run_name': "ablation:sms_wsj_run_spatial_net",

    #saving
    'model_save_path': "/home/clem3nti/projects/SpatialSoundMix/checkpoints", 
    'resume': True, 
    'model_load_path': "/home/clem3nti/projects/SpatialSoundMix/checkpoints/train_state.pth", 
    
    #set testing mode for only running inference
    'test': False, 
    'test_write_path': "/home/clem3nti/projects/SpatialSoundMix/test_audio", 
    'test_model_path': "/home/clem3nti/projects/SpatialSoundMix/checkpoints/train_state.pth",
}

def train(config): 
    print(f"device {config['device']}")
    model = SpatialNet(
        config['num_mics'] * 2, 
        config['num_speakers'] * 2, 
        config['dim_squeeze'], 
        config['num_layers'], 
        config['num_freqs'], 
        num_heads=config['num_heads'], 
        checkpointing = config['checkpointing'], 
    ).to(config['device'])
    print("model initialized")


    print("training functions initialized")

    train_dataset = SmsWsjDataset(
        config['wsj_root'], 
        config['target'], 
        "train_si284", 
        audio_time_len=config['audio_time_len'], 
        num_spk=config['num_speakers']
    )
    val_dataset = SmsWsjDataset(
        config['wsj_root'], 
        config['target'], 
        "cv_dev93", 
        audio_time_len=config['audio_time_len'], 
        num_spk=config['num_speakers']
    )
    inf_dataset = SmsWsjDataset(
        config['wsj_root'], 
        config['target'], 
        "test_eval92", 
        audio_time_len=None, 
        num_spk=config['num_speakers']
    )

    print("datasets loaded")

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config['batch_size_train'],
        num_workers=config['num_workers'], 
        persistent_workers=config['persistent_workers'], 
        pin_memory=config['pin_memory'], 
        prefetch_factor=config['prefetch_factor'], 
        shuffle=True
    )

    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size=config['batch_size_val'],
        num_workers=config['num_workers'], 
        persistent_workers=config['persistent_workers'], 
        pin_memory=config['pin_memory'], 
        prefetch_factor=config['prefetch_factor'], 
        shuffle=False
    )

    inf_loader = DataLoader(
        dataset = inf_dataset,
        batch_size=config['batch_size_inf'],
        num_workers=config['num_workers'], 
        persistent_workers=config['persistent_workers'], 
        pin_memory=config['pin_memory'], 
        prefetch_factor=config['prefetch_factor'], 
        shuffle=False
    )

    print("dataloaders initialized")

    loss = factorial_SA_SDR(config['num_speakers'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # lr_scheduler = lr.CosineAnnealingLR(optimizer, config['epochs'])
    total_steps = config['epochs'] * len(train_loader) // config['grad_acum_steps']
    lr_scheduler = lr.OneCycleLR(
        optimizer, 
        max_lr=config['lr'], 
        total_steps=total_steps, 
        pct_start=config['pct_start'], 
        div_factor=config['div_factor'],
        final_div_factor=config['final_div_factor'], 
    )


    trainer = WsjSpeechSeparationTrainer(
        model, config['num_speakers'], 
        config['stft_parameters'], optimizer, loss, 
        lr_scheduler, config['batch_scheduler'],
        train_loader, val_loader, inf_loader, 
        config['inf_chunksize'], config['inf_hopsize'], 
        config['grad_acum_steps'], config['wandb_key'], 
        sample_rate = config['d_sample_rate'], 
        mixed_precision=config['mixed_precision'], 
        use_wandb=config['use_wandb'], 
        device=config['device'],
    )

    if config['test']: 
        print("running test separations")
        trainer.load_train_state(config['test_model_path'])
        trainer.validate_epoch()
        trainer.validate_full_separations(
            write_separations=True, 
            separations_write_path=config["test_write_path"]
        )

    else: 
        print("trainer initialized beginning training")
        trainer.train(
            config['epochs'], 
            config, 
            config['model_save_path'], 
            config['run_name'],
            resume = config['resume'], 
            model_load_path = config['model_load_path'], 
            full_val_steps = config['full_val_steps'], 
        )

if __name__ == "__main__": 
    train(config)