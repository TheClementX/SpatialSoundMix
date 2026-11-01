"""
datasets.py

SpatialSoundMixStaticDataset

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
import torch.utils.data.Dataset as Dataset

"""
@brief: This dataset is meant to be used when a SpatialSoundMix instance
has been generated via scene. This is when each mix has already been made
with static positioning per instance. An instance is denoted by a choice of 
speakers, speaker positions, and a microphone position. 

@pre: assumes that mix and souruces are same length for slicing
"""
class SpatialSoundMixStaticDataset(Dataset): 
    def __init__(
        self, 
        data_root: str, 
        clip_length: int = 4,
        random_slices: bool = True,
        sample_rate: float = 44100.0
    ) -> None: 
        self.data_root = data_root
        self.clip_length = clip_length
        self.random_slices = random_slices
        self.sample_rate = sample_rate 

        self.slice_length = self.sample_rate * self.clip_length

        self.scene_paths = [
            os.path.join(self.data_root, sp) for sp in os.listdir(self.data_root)
        ]

        self.instance_paths = []
        for sp in self.scenes: 
            self.instance_paths.extend(
                [sp + instance for instance in os.listdir(sp)]
            )

    def __len__(self) -> int: 
        return len(self.instance_paths)

    def __getitem__(
        self, 
        idx: int
    ) -> dict[str, torch.Tensor]:
        ip = self.instance_paths[idx]

        #(samples, channels)
        mix_audio, mix_sr = sf.load(ip + "mix_audio.wav")
        #(samples, channels) -> (channels, samples)
        mix_audio = mix_audio.T
        sources = []
        for file in os.listdir(ip): 
            if file.startswith("source"): 
                sp = ip + file
                s, sr = sf.load(sp)
                sources.append(s.T)

        #select random start
        start_idx = 0
        if self.random_slices: 
            audio_len = mix_audio.shape[0]
            if audio_len < self.slice_length: 
                raise ValueError("audio example to short and underflowed window length")

            max_len = audio_len - self.slice_length
            start_idx = random.randint(0, max_len)

        end_idx = start_idx + self.slice_length
        mix_audio_slice = mix_audio[start_idx:end_idx]
        sources = torch.stack(sources)

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
    def __init__(self) -> None: 
        pass

    def __len__(self) -> int: 
        pass

    def __getitem__(self, idx: int) -> torch.Tensor: 
        pass