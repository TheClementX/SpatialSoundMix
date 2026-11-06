#local imports
from models.losses import MAELoss
from models.spatialnet.SpatialNet import * 
from training.datasets import *
from training.trainer import SpeechSeparationTrainer

#pytorch imports
import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr 


config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", 
    #model 
    'num_speakers': 4, 
    'num_mics': 2, 
    'dim_squeeze': 8, 
    'num_layers': 8, 
    'num_freqs': 129, 
    "stft_parameters": {
        "n_fft": 256,
        "hop_length": 128,
        "win_length": 256,
        "in_channels": 2, #mixes are binaural
        "out_channels": 1, #separations are monaural
        "window": "hann"
    }, 

    #datasets
    "train_root": "/home/clem3nti/projects/SpatialSoundMix/spatial_data/train", 
    "val_root": "/home/clem3nti/projects/SpatialSoundMix/spatial_data/val", 
    "test_root": "/home/clem3nti/projects/SpatialSoundMix/spatial_data/test", 
    "clip_length": 1, 
    "random_slices": True, 
    "sample_rate": 44100, 

    #training
    'lr': 5e-4, 
    'batch_size_train': 1, 
    'batch_size_val': 4, 
    'batch_size_inf': 1,
    'sample_rate': 44100,
    'epochs': 10, 
    'warmup_epochs': None, 
    'grad_acum': True, 
    'grad_acum_steps': 2, 
    'mixed_precision': True, 
    'batch_scheduler': False, 

    #dataloaders
    'num_workers': 12, 
    'pin_memory': True,
    'persistent_workers': False, 
    'prefetch_factor': 4,

    #inference
    "inf_chunksize": 441000, 
    "inf_hopsize": 220500, 

    #wandb
    'wandb_key': "", 
    'use_wandb': False, 

    #saving
    'model_save_path': "/home/clem3nti/projects/SpatialSoundMix/checkpoints", 
    'run_name': "ablation:1",
}

def train(config): 
    model = SpatialNet(
        config['num_mics'] * 2, 
        config['num_speakers'] * 2, 
        config['dim_squeeze'], 
        config['num_layers'], 
        config['num_freqs'], 
    ).to(config['device'])
    print("model initialized")

    loss = MAELoss() #using defaults
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    lr_scheduler = lr.CosineAnnealingLR(optimizer, config['epochs'])

    print("training functions initialized")

    train_dataset = SpatialSoundMixStaticDataset(
        config['train_root'], 
        config['num_speakers'], 
        clip_length=config['clip_length'], 
        random_slices=config['random_slices'], 
        sample_rate = config['sample_rate'] 
    )
    
    val_dataset = SpatialSoundMixStaticDataset(
        config['val_root'], 
        config['num_speakers'], 
        clip_length=config['clip_length'], 
        random_slices=config['random_slices'], 
        sample_rate = config['sample_rate'] 
    )

    inf_dataset = SpatialSoundMixInferenceDataset(
        config['test_root'], 
        config['num_speakers']
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


    trainer = SpeechSeparationTrainer(
        model, config['num_speakers'], 
        config['stft_parameters'], optimizer, loss, 
        lr_scheduler, config['batch_scheduler'],
        train_loader, val_loader, inf_loader, 
        config['inf_chunksize'], config['inf_hopsize'], 
        config['grad_acum_steps'], config['wandb_key'], 
        mixed_precision=config['mixed_precision'], 
        use_wandb=config['use_wandb'], 
        device=config['device'],
    )

    print("trainer initialized beginning training")

    trainer.train(
        config['epochs'], 
        config, 
        config['model_save_path'], 
        config['run_name'],
    )

if __name__ == "__main__": 
    train(config)