"""
datasets.py

SpatialSoundMixStaticDataset

SpatialSoundMixInferenceDataSet

data_root structure: 
    --> root
      |--> train
        |--> scene_i
          |--> instance_i
            |--> mix_audio.wav
            |--> source_i.wav
            |--> counts.npy

SpatialSoundMixDynamicDataset
data_root structure: 
    --> root
      |--> train
        |--> scene_i
          |--> instance_i
            |--> speaker_i.wav

"""

import torch
import os 
import soundfile as sf 
import random
from torch.utils.data import Dataset
import torchaudio.transforms as T

"""
@brief: This dataset is meant to be used when a SpatialSoundMix instance
has been generated via scene. This is when each mix has already been made
with static positioning per instance. An instance is denoted by a choice of 
speakers, speaker positions, and a microphone position. 

@pre: assumes that mix and sources are same length for slicing
"""
class SpatialSoundMixStaticDataset(Dataset): 
    def __init__(
        self, 
        data_root: str, 
        num_speakers: int, 
        clip_length: int = 4,
        random_slices: bool = True,
        sample_rate: int = 44100, 
        d_sample_rate: int = 44100, 
    ) -> None: 
        self.data_root = data_root
        self.clip_length = clip_length
        self.random_slices = random_slices
        self.sample_rate = sample_rate 
        self.d_sample_rate = d_sample_rate
        self.num_speakers = num_speakers

        self.slice_length = self.sample_rate * self.clip_length

        self.scene_paths = [
            os.path.join(self.data_root, sp) for sp in os.listdir(self.data_root)
        ]

        self.instance_paths = []
        for sp in self.scene_paths: 
            self.instance_paths.extend(
                [os.path.join(sp, instance) for instance in os.listdir(sp)]
            )
        self.instance_paths = sorted(self.instance_paths)
        self.resampler = T.Resample(orig_freq=sample_rate, new_freq=d_sample_rate)

    def __len__(self) -> int: 
        return len(self.instance_paths)

    """
    @return 
        mix_audio_slice: (B, chan, samples) chan = 2 for Spatial Soundmix
        sources: (B, num_speakers, samples)
    """
    def __getitem__(
        self, 
        idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ip = self.instance_paths[idx]

        #(samples, channels)
        mix_audio, mix_sr = sf.read(os.path.join(ip, "mix_audio.wav"))
        #(samples, channels) -> (channels, samples)
        mix_audio = torch.from_numpy(mix_audio.T).float()

        #select random start
        start_idx = 0
        if self.random_slices: 
            audio_len = mix_audio.shape[1]
            if audio_len < self.slice_length: 
                raise ValueError("audio example to short and underflowed window length")

            max_len = audio_len - self.slice_length
            start_idx = random.randint(0, max_len)

        end_idx = start_idx + self.slice_length
        mix_audio_slice = mix_audio[:, start_idx:end_idx]

        sources = []
        for file in os.listdir(ip): 
            if file.startswith("source"): 
                sp = os.path.join(ip, file)
                s, sr = sf.read(sp)
                sources.append(torch.from_numpy(s[start_idx:end_idx]).float())

        num_silent = self.num_speakers - len(sources)
        for _ in range(num_silent): 
            sources.append(torch.zeros(self.slice_length))
        sources = torch.stack(sources, dim=0)

        #downsample
        if self.d_sample_rate != self.sample_rate: 
            mix_audio_slice = self.resampler(mix_audio_slice)
            sources = self.resampler(sources) 
        #return slices
        return mix_audio_slice, sources


"""
@brief: This dataset is meant for use as a random mixing dataset. 
In this scenario each scene contains one recording convolved with a 
RIR from 1 microphone position per speaker in a subset. This 
structure is meant for random mixing. 
"""
class SpatialSoundMixDynamicDataset(Dataset): 
    def __init__(self) -> None: 
        pass

    def __len__(self) -> int: 
        pass

    def __getitem__(self, idx: int) -> torch.Tensor: 
        pass


"""
@brief: a simple dataset for storing instances for inference
"""
class SpatialSoundMixInferenceDataset(Dataset): 
    def __init__(
        self, 
        data_root: str, 
        num_speakers: int
    ) -> None: 
        self.data_root = data_root
        self.num_speakers = num_speakers

        self.scene_paths = [
            os.path.join(self.data_root, sp) for sp in os.listdir(self.data_root)
        ]

        self.instance_paths = []
        for sp in self.scene_paths: 
            self.instance_paths.extend(
                [os.path.join(sp, instance )for instance in os.listdir(sp)]
            )
        self.instance_paths = sorted(self.instance_paths)

    def __len__(self) -> int: 
        return len(self.instance_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ip = self.instance_paths[idx]

        #(samples, channels)
        mix_audio, mix_sr = sf.read(os.path.join(ip, "mix_audio.wav"))
        #(samples, channels) -> (channels, samples)
        mix_audio = torch.from_numpy(mix_audio.T).float()

        sources = []
        for file in os.listdir(ip): 
            if file.startswith("source"): 
                sp = os.path.join(ip, file)
                s, sr = sf.read(sp)
                sources.append(torch.from_numpy(s).float())

        num_silent = self.num_speakers - len(sources)
        for _ in range(num_silent): 
            sources.append(torch.zeros(len(sources[0])))
        sources = torch.stack(sources, dim=0)

        return mix_audio, sources