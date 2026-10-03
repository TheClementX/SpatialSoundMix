"""
static_dataset_builder.py

@brief: This file contains the necessary classes to build the static 
VoiceSpatialSoundMix Dataset. 

habitat-sim: habitat-sim is an audio-engine provided by meta research 
for calculating RIRs using ray tracing from an Agent to some amount of 
Audio Sources. The Audio Sources do not store audio. Instead the simulator
can be thought of as containing all the information necessary to build 
a spatialized mix of audio with n sources with any type of input audio. 

approach: The general idea is to create a variety of audio scapes using a
varying number of Audio Sources and HM3D Scenes. For each HM3D scene
a couple of different configurations of numbers of AudioSensors will be created. 
For each variation differing audio tracks will be played in each AudioSensors
to create a diverse set of speaker data for that scene and AudioSensor configuration. 
Furthermore the data for audio scape will be split into stationary and dynamic
data. Stationary data will contain AudioSensors and Receivers that do not move. 
Dynamic data will contain AudioSensors and Receivers that move in smooth trajectories
throughout the HM3D scene. We will define the generation steps in pseudo code bellow. 

DATA GENERATION STEPS: 

    for scene in HM3D dataset scenes: 
        for some fixed amount of AudioSensor and Reciever configurations: 
            for some number of overlapping speaker sequences: 
                1.) generate stationary data corresponding to the Audio Scape
                2.) generate dynamic data corresponding to teh Audio Scape 


Audio Rendering: In general simulation will be broken down into some number of steps s. 
for each step a certain amount of samples will be played in the current audio scape 
configuration. These samples will be convolved with the generated habitat-sim RIR and then
overlap and add will be used to record them into a final audio array which will be written to 
disk. 
"""

import os
import habitat_sim
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import random

"""
AudioCue

@brief: Used in a list to queue audio objects for playing. 
an AudioCue stores its audio and some metadata. an AudioCue
stores its start step which is used for sequencing in and AudioSequence.
"""
class AudioCue: 
    def __init__(
        self,
        raw_audio_path: str, 
        sensor_name: str, 
        sample_rate: float = 44100.0
    ) -> None: 
        self.path = raw_audio_path
        self.audio, self.sample_rate = sf.read(raw_audio_path)
        self.sensor_name = sensor_name
        self.num_samples = len(self.audio)

    def set_cue_start_step(self, start_step int): 
        self.start_step = start_step

"""
AudioSequence

@breif: The main class for managing samples and the timing 
of audio playing. This is an indexable object which can 
be used to return the audio to be convolved with a RIR
at simulation step i. 
"""
class AudioSequence: 
    def __init__(
        self,
        cues: list[AudioCue], 
        max_samples: int, 
        samples_per_step: int, 
    ) -> None: 
        self.cues = cues
        self.max_samples = max_samples
        self.samples_per_step = samples_per_step
        self.max_steps = (max_samples // samples_per_step) + 1

    """
    @brief: return the samples for the current time step for each 
    active audio source. 
    """
    def __getitem__(self, time: int) -> list[tuple[str, np.ndarray]]:
        result = []
        for i, q in enumerate(self.cues): 
            if time <= q.start_step: 
                start = (time - q.start_step) * samples_per_step
                end = start + self.samples_per_step
                cur_audio = q.audio[start:end]
                result.append((f"audio_source_{i}", cur_audio))

        return result

"""
SimState

@brief: Used to record the current state of the simulation
"""
class SimState: 
    def __init__(): 
        pass

"""
StaticBSCDatastBuilder

@breif: The class used for creating the full SpatialSoundMix
train dataset. 
"""
class SpatialSoundMixDatasetBuilder: 
    """
    __init__

    @brief: load all mesh and audio source paths. 
    """
    def __init__(
        data_dir: str, 
        mesh_dir: str, 
        fps: int, 
        sample_rate: float = 44100.0,
        seed: int = 42,
        data_type: str = "EARS"
    ) -> None: 
        self.data_dir = data_dir 
        self.data_paths = [
            os.path.join(self.data_dir, dir) for dir in os.listdirs(data_dir)
        ]

        self.mesh_dir = mesh_dir
        self.mesh_paths = [
            os.path.join(self.mesh_dir, dir) for dir in os.listdirs(mesh_dir)
        ]

        self.seed = seed

    """
    set_random_elevation

    @brief: add elevation to a point. This simulates the height of a speaker. 
    """
    def set_random_elevation(
        sim: habitat_sim.simulator.Simulator, 
        point: np.ndarray, 
        default_height: float = 1.6, 
        min_height: float = 0.05, 
        ceiling_offset: float = 0.05
    ) -> np.ndarray: 
        
        #elevate point and set direction for ray cast
        point = point + np.ndarray([0.0, 0.05, 0.0])
        ray_direction = np.ndarray([0.0, 1.0, 0.0])

        #cast array using physics engine
        ray = habitat_sim.geo.Ray(ray_origin, ray_direction)
        hit_record = sim.cast_ray(ray)

        if hit_record.has_hits: 
            distance_to_ceiling = hit_record.hits[0].hit_distance
            max_height = distance_to_ceiling - ceiling_offset
        else: 
            max_height = minimum_height

        #sample point
        random_elevation = random.uniform(min_height, max_height)

        new_point = point.copy()
        new_point[1] += random_elevation

        return new_point

    """
    set_random_audio_source_position
    
    @brief: sets the choosen AudioSensor position in the current
    simulation to a random navigable and valid position.@brief: 
    """
    def set_random_audio_source_position(): 
        pass 

    """
    @brief: Initializes the habitat-sim simulation state with a scene and 
    number of sources

    TODO: change audio_source_idxes -> number_of_sources. Audio sources will be 
    independent from habitat-sim context. 
    """
    def initialize_sim_scene(
        audio_source_idxes: list[int]
        mesh_idx: int
    ) -> habitat_sim.simulator.Simulator: 
        #setup Backend and scene
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = sShould I set the location of audio souelf.mesh_paths[mesh_idx]
        backend_cfg.enable_physics = True
        backend_cfg.random_seed = self.seed

        #setup acoustics
        acoustics_cfg = habitat_sim.RLRAudioPropagationConfiguration()
        acoustics_cg.enableMaterials = True
        acoustics_cfg.temporalCoherence = True

        #setup audio sources
        sensor_specs = []
        for i, idx in enumerate(audio_source_idxes): 
            audio_spec = habitat_sim.AudioSensorSpec()
            audio_spec.uuid = f"audio_source_{i}"
            audio_spec.acousticsConfig = acoustics_cfg

            #binaural setup
            audio_spec.channelLayout.channelType = (
                habitat_sim.RLRAudioPropagationChannelLayoutType.Binaural
            )
            sensor_specs.append(audio_spec)

        agent_cfg = habitat_sim.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

        sim = habitat_sim.Simulator(cfg)
        agent = sim.get_agent(0)

        #set initial audio source positions
        for i in range(len(audio_sources_idxes)): 
            valid_floor_pos = sim.pathfinder.get_random_navigable_point()
            initial_pos = self.set_random_elevation(valid_floor_pos)
            sensor = agent._sensors[f"audio_source+{i}"]
            sensor.setAudioSourceTransform(valid_floor_pos)

        sim.step({})
        return sim

    """
    create_audio_queue_sequence

    @brief: sequences audio in time randomely to make 
    a sequence of AudioCues.
    @return: An AudioSequence object used for get the 
    samples for convolution at each timestep. 
    """
    def create_audio_cue_sequence(
        self, 
        raw_audio_paths: list[str],
        step_duration: float = 0.2, 
        additional_seconds: float = 5.0
    ) -> AudioSequence: 
        cues, source_max_samples = [], 0
        sample_rate = None

        #load AudioCues
        for i, path in enumerate(raw_audio_paths): 
            cues.append(AudioCue(path, f"audio_source_{i}"))
            if i == 0: 
                sample_rate = cues[0].sample_rate
            if cues[i].sample_rate != sample_rate: 
                raise ValueError("Sample rate mismatch in audiosequence")
            source_max_samples = max(cues[i].num_samples, source_max_samples)

        additional_samples = additional_seconds * sample_rate
        samples_per_step = step_duration * sample_rate
        sequence_samples = source_max_samples + additional_samples
        #make sure sequence samples is multiple of sequence_steps
        sequence_samples += samples_per_step - (sequence_samples % samples_per_step)
        sequence_steps = sequence_samples // samples_per_step

        #select audio source start times
        #TODO: finalize start times
        for q in cues: 
            start = additional_samples // 2


        sequence = AudioSequence(cues, sequence_samples,  samaples_per_step)
        return sequence

    """
    create_audio_trajectories

    @brief: creates the trajectories for AudioSensors in habitat-sim
    """
    def create_audio_trajectories(): 
        pass

    """
    create_receiver_trajectories

    @brief: creates the trajectories for the agent / reciever in habitat-sim
    """
    def create_receiver_trajectories(): 
        pass 

    """
    generate_stationary_data_instace

    @breif: Generates one instance of simulated audio with static positioning. 

    @param: 
    @return: 
    """
    def generate_stationary_data_instance(
        audio_cues: list[AudioCue],  
    ) -> dict: 
        pass

    """
    generate_dynamic_data_instances

    @brief: Generates one instance of a dynamic data point using
    audio sensor paths and receiver paths. 
    """
    def generate_dynamic_data_instance(
        audio_queues: list[AudioQueue], 
        audio_sensor_paths: list[np.ndarray], 
        receiver_path: np.ndarray
    ) -> dict: 
        pass

    """
    generate_static_instance_for_scene

    @brief: generates all the static instances for the current audio space
    """
    def generate_static_instances_for_scene(): 
        pass

    """
    generate_dynamic_instances_for_scene

    @brief: generates all dynamic instance of audio for a scene
    """
    def generate_dynamic_instances_for_scene(): 
        pass

    """
    build_dataset

    @breif: builds the whole train dataset
    """
    def build_dataset(): 
        pass


