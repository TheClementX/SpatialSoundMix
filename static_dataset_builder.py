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
                2.) generate dynamic data corresponding to the Audio Scape 


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
import scipy.signal as ss 
import random
from tqdm import tqdm 
import gc

"""
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
        self.audio = self.audio 
        self.sensor_name = sensor_name
        self.num_samples = len(self.audio)
        if sample_rate != 44100.0: 
            raise ValueError("Sample rate not 44100.0 in AudioCue instantiation")
        self.sample_rate = sample_rate
        self.activity_mask = self.create_activity_array(self.audio)

    def set_cue_start_step(self, start_step: int) -> None: 
        self.start_step = start_step

    """
    a function to compute exactly when a slice of audio is playing vs 
    silent. Use the hilbert transform approach.  

    #TODO: mathematically check the validity of this activity counting 
    approach

    @return: a binary mask where 1 corresponds to an active signal
    source and 0 corresponds to no source. 
    """
    def create_activity_array(
        self, 
        signal: np.ndarray, 
        threshold: float = 0.01
    ) -> np.ndarray: 
        analytic_signal = ss.hilbert(signal)
        envelope = np.abs(analytic_signal)
        activity_mask = (envelope > threshold).astype(int)

        return activity_mask 

"""
@brief: The main class for managing samples and the timing 
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
            if time >= q.start_step: 
                start = (time - q.start_step) * self.samples_per_step
                end = start + self.samples_per_step
                cur_audio = q.audio[start:end]
                cur_mask = q.activity_mask[start:end]
                result.append((f"audio_source_{i}", cur_audio, cur_mask))

        return result

    def __len__(self): 
        return self.max_steps

"""
@breif: The class used for creating the full SpatialSoundMix
train dataset. 
"""
class SpatialSoundMixDatasetBuilder: 
    """
    __init__

    @brief: load all mesh and audio source paths. 
    """
    def __init__(
        self, 
        data_dirs: list[str], 
        mesh_dirs: list[str], 
        audio_absorption_file: str,
        dataset_config_file: str, 
        step_duration: float, 
        sample_rate: float = 44100.0,
        seed: int = 42,
        filter_by_semantic_scene: bool = True
    ) -> None: 

        self.audio_absorption_file = audio_absorption_file
        self.dataset_config_file = dataset_config_file

        self.data_dirs = data_dirs
        self.data_paths = []
        for p_dir in self.data_dirs: 
            self.data_paths.extend([
                os.path.join(p_dir, dir) for dir in os.listdir(p_dir)
            ])

        self.mesh_dirs = mesh_dirs
        #this is a list of the directories containing specific meshes
        #filter out non mesh dirs
        self.mesh_paths = [] 
        for p_dir in self.mesh_dirs:
            for dir in os.listdir(p_dir): 
                path = os.path.join(p_dir, dir)
                if not os.path.isdir(path): continue
                #check for semantics files if necessary
                if (
                    filter_by_semantic_scene 
                    and not self.scene_contains_semantics(path)
                ): continue

                self.mesh_paths.append(path)

        self.step_duration = step_duration
        self.sample_rate = sample_rate
        self.seed = seed

    def scene_contains_semantics(self, scene_path: str) -> bool:
        scene_files = os.listdir(scene_path)
        path_base = os.path.basename(scene_path).split("-")[-1]
        semantic_name = path_base + ".semantic.glb"
        for f in scene_files: 
            if f == semantic_name: return True
        return False



    """
    @brief: add elevation to a point. This simulates the height of a speaker. 
    @param: 
        sim: the habitat-sim instance
        point: (3,) the point to add elevation to 
        default_height: the default height for unbounded rooms
        min_height: the minimum allowed height
        ceiling_offset: the amount to depress a point from the ceiling
    @return: 
        a new elevated point    
    """
    def set_random_elevation(
        self, 
        sim: habitat_sim.simulator.Simulator, 
        point: np.ndarray, 
        default_height: float = 1.6, 
        min_height: float = 0.05, 
        ceiling_offset: float = 0.05
    ) -> np.ndarray: 
        
        #elevate point and set direction for ray cast
        point = np.asarray(point, dtype=np.float32) + np.array([0.0, 0.05, 0.0])
        ray_direction = np.array([0.0, 1.0, 0.0])

        #cast array using physics engine
        ray = habitat_sim.geo.Ray(point, ray_direction)
        hit_record = sim.cast_ray(ray)

        if len(hit_record.hits) > 0: 
            distance_to_ceiling = hit_record.hits[0].ray_distance
            max_height = max(min_height, distance_to_ceiling - ceiling_offset)
        else: 
            max_height = default_height

        #sample point
        random_elevation = random.uniform(min_height, max_height)
        new_point = point.copy()
        new_point[1] += random_elevation

        return new_point

    """
    @brief randomly set the AudioSensors in a sim object
    to valid locations.

    @param: 
        sim: the simulation whos audio sensor positions will 
        be set.
        num_sources: the number of audio sources associated 
        with the sim object.
    """
    def set_random_audio_source_positions(
        self,
        sim: habitat_sim.simulator.Simulator,
        num_sources: int,
        island_index: int, 
        max_distance: float = float("inf") 
    ) -> None: 
        #set position of agent to random point
        agent = sim.get_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_pos = sim.pathfinder.get_random_navigable_point(island_index=island_index)
        agent_state.position = agent_pos
        agent.set_state(agent_state)

        #set position of sensors to random points
        for i in range(num_sources): 
            valid_floor_pos = sim.pathfinder.get_random_navigable_point_near(
                circle_center=agent_pos,
                radius=max_distance,
                max_tries=400,
                island_index=island_index
            )
            path = habitat_sim.ShortestPath()
            path.requested_start = agent_pos
            path.requested_end = valid_floor_pos
            found_path = sim.pathfinder.find_path(path)

            if not found_path: 
                valid_floor_pos = sim.pathfinder.get_random_navigable_point(
                    island_index=island_index
                )

            initial_pos = self.set_random_elevation(sim, valid_floor_pos)
            sensor = agent._sensors[f"audio_source_{i}"]
            sensor.setAudioSourceTransform(initial_pos)


        #warmup simulator to new positions
        sim.step({})


    def get_largest_island_for_scene(
        self, 
        sim: habitat_sim.simulator.Simulator
    )-> int: 
        pf = sim.pathfinder 
        areas = [pf.island_area(i) for i in range(pf.num_islands)]
        largest_index = int(np.argmax(areas))

        return largest_index

    """
    @brief: Initializes the habitat-sim simulation state with a scene and 
    number of sources

    @param: 
        num_sources: the number of AudioSensors to associate with the 
        new scene
        mesh_path: the path to the .glb file 
    @return: 
        the new simulator
    """
    def initialize_sim_scene(
        self,
        num_sources: int, 
        mesh_path: str,
        sim: habitat_sim.Simulator = None,
        max_distance: float = 10.0
    ) -> habitat_sim.simulator.Simulator: 
        #setup Backend and scene
        print(f"loading scene with {num_sources} sources at : {mesh_path}")
        backend_cfg = habitat_sim.SimulatorConfiguration()

        #get scene handle not full file path
        backend_cfg.scene_dataset_config_file = self.dataset_config_file
        backend_cfg.scene_id = mesh_path.split("hm3d/")[-1]
        backend_cfg.load_semantic_mesh = True
        #this line was the fix to our bug
        #the habitat-sim backend was misinterpreting semantic.glb as geometry
        backend_cfg.use_semantic_textures = False 
        backend_cfg.enable_physics = True
        backend_cfg.random_seed = self.seed

        #setup acoustics
        acoustics_cfg = habitat_sim.RLRAudioPropagationConfiguration()
        acoustics_cfg.enableMaterials = True 
        acoustics_cfg.temporalCoherence = False

        #setup audio sources
        sensor_specs = []
        for i in range(num_sources): 
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

        if sim is None: 
            sim = habitat_sim.Simulator(cfg)
        else: 
            sim.reconfigure(cfg)
        sim.seed(self.seed)

        if len(sim.semantic_scene.objects) == 0: 
            print("semantics not loaded properly in scene initialization")
            raise ValueError("Semantics not loaded")

        #initialize navmesh
        navmesh_path = mesh_path[:-4] + ".navmesh"
        nav_mesh_exit = sim.pathfinder.load_nav_mesh(navmesh_path)

        if not nav_mesh_exit: 
            raise ValueError("attempted loading an invalid navmesh")

        #load acoustic file for each sensor 
        agent = sim.get_agent(0)
        for sensor_name, sensor in agent._sensors.items(): 
            #only audio sensors are made so only audio sensors are accessed
            sensor.setAudioMaterialsJSON(self.audio_absorption_file)

        #set initial sim positions
        island_idx = self.get_largest_island_for_scene(sim)
        self.set_random_audio_source_positions(
            sim, num_sources, island_idx, max_distance=max_distance
        )
        print("scene initialized")

        return sim

    """
    @brief: sequences audio in time randomely to make 
    a sequence of AudioCues.
    @return: An AudioSequence object used for get the 
    samples for convolution at each timestep. 
    """
    def create_audio_cue_sequence(
        self, 
        raw_audio_paths: list[str],
        step_duration: float = 0.2, 
        additional_seconds: float = 10.0
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

        #caclulate additional seconds 
        additional_samples = int(additional_seconds * sample_rate)
        padding = additional_samples // 2

        samples_per_step = int(step_duration * sample_rate)
        sequence_samples = int(source_max_samples + additional_samples)

        #make sure sequence samples is multiple of sequence_steps
        sequence_samples += samples_per_step - (sequence_samples % samples_per_step)

        #select audio source start times
        for q in cues: 
            start = padding
            end = sequence_samples - padding - q.num_samples

            sample_idx = random.randint(start, end)
            start_step = sample_idx // samples_per_step
            q.set_cue_start_step(start_step)

        sequence = AudioSequence(cues, sequence_samples, samples_per_step)
        return sequence, sample_rate

    """
    @brief:
        creates a list of points corresponding to a 
        valid navigable path that an AudioSensor or Reciever 
        can take. 
    @params

    """
    def create_navigable_trajectory(
        sim: habitat_sim.simulator.Simulator, 
        start_point: np.ndarray, 
        num_steps: int, 
        distance_per_step: float, 
        other_trajectories: list[list[np.ndarray]]
    ) -> list[np.ndarray]: 
        pass

    """
    @brief: creates the trajectories for AudioSensors in habitat-sim
    """
    def create_audio_trajectories(self): 
        pass

    """
    @brief: creates the trajectories for the agent / reciever in habitat-sim
    """
    def create_receiver_trajectories(self): 
        pass 


    """
    @breif: Generates one instance of simulated audio with static positioning. 
    The current setup is for generating BSC data but can be easily altered
    by simply returning the target audio separation as well. 

    @param: 
        sim: the habit_sim simulator for ray tracing
        audio_sequence: the audio sequence for enumerating time slices
        num_channels: the number of channels of the sim
    @return: 
    """
    def generate_stationary_data_instance(
        self, 
        sim: habitat_sim.simulator.Simulator,
        audio_sequence: AudioSequence, 
        num_channels: int = 2
    ) -> dict: 
        mix_audio = np.zeros((num_channels, audio_sequence.max_samples))
        source_counts = np.zeros((num_channels, audio_sequence.max_samples))

        prog_bar = tqdm(
            range(len(audio_sequence)), 
            desc="Generating Sequence", 
            leave=False
        )

        #only get observations once for stationary audio
        obs = sim.get_sensor_observations()
        for i in prog_bar: 
            #calculate current RIRs
            audio_slices = audio_sequence[i]

            #convolve RIRs for current time step 
            for s in audio_slices: 
                #s_audio (channels, samples)
                s_name, s_audio, s_mask = s
                if len(s_audio) == 0: 
                    continue # don't process empty audio

                s_start = i * audio_sequence.samples_per_step
                s_end = min(s_start + len(s_mask), audio_sequence.max_samples)
                counts_len = s_end - s_start

                rir = np.array(obs[s_name])

                left_reciever_channel = ss.fftconvolve(s_audio, rir[0, :], mode="full")
                right_reciever_channel = ss.fftconvolve(s_audio, rir[1, :], mode="full")

                #increment counts
                source_counts[0, s_start:s_end] += s_mask[:counts_len]
                source_counts[1, s_start:s_end] += s_mask[:counts_len]

                #calculate ending positions adjusting for reverb tails
                left_len = len(left_reciever_channel)
                mix_end_left = min(s_start + left_len, audio_sequence.max_samples)
                left_len = mix_end_left - s_start

                right_len = len(right_reciever_channel)
                mix_end_right = min(s_start + right_len, audio_sequence.max_samples)
                right_len = mix_end_right - s_start

                #overlap add audio
                mix_audio[0, s_start:mix_end_left] += left_reciever_channel[:left_len]
                mix_audio[1, s_start:mix_end_right] += right_reciever_channel[:right_len]

        #normalize audio mix after generation
        amp_max = np.max(np.abs(mix_audio))
        if amp_max > 0: 
            mix_audio /= amp_max

        #generate source audio : Sources are mono
        sources = []
        for q in audio_sequence.cues: 
            #prepend silence
            audio = q.audio
            pre_silence_len = q.start_step * audio_sequence.samples_per_step
            pre_silence = np.zeros(pre_silence_len)
            p_audio = np.concatenate([pre_silence, audio], axis=0)

            #postpend silence
            post_silence_len = audio_sequence.max_samples - len(p_audio)
            post_silence = np.zeros(post_silence_len)
            pre_post_audio = np.concatenate([p_audio, post_silence], axis=0)
            sources.append(pre_post_audio)

        return {
            "mix_audio": mix_audio, 
            "source_counts": source_counts,
            "sources": sources
        }


    """
    @brief: Generates one instance of a dynamic data point using
    audio sensor paths and receiver paths. 
    """
    def generate_dynamic_data_instance(
        self, 
        audio_queues: AudioSequence, 
        audio_sensor_paths: list[np.ndarray], 
        receiver_path: np.ndarray
    ) -> dict: 
        pass

    """
    @brief: generates all the static instances for the current hm3d scene. 
    This writes all the audio as well and builds the simulator for the
    current scene.

    @param: 
        mesh_dir: the directory containing the scene and nav mesh
        num_sensors: the number of AudioSensors for this scene
        num_states: the number of different position configurations to 
        generate
        instance_name: the name for writig
        data_write_path: where the new scene data folder will be written
        step_duration: how many seconds are processed at once
        mesh_type: the type of mesh to load
    @note: destroys the scene before returning
    """
    def generate_stationary_instances_for_scene(
        self,
        mesh_dir: str, 
        num_sensors: int, 
        num_states: int, 
        instance_name: str,
        data_write_path: str, 
        return_sources: bool = True, 
        step_duration: float = 0.2, 
        mesh_type: str = "glb", 
        sim: habitat_sim.simulator.Simulator = None,
        max_distance: float = 10.0
    ) -> None: 
        #parse mesh path 
        mesh_path = ""
        if mesh_type == "glb": 
            for p in os.listdir(mesh_dir): 
                if p.endswith(".basis.glb"): 
                    mesh_path = os.path.join(mesh_dir, p)
                    break
        else: 
            raise ValueError("only .glb type meshes supported for simulation")

        print("checking mesh path")
        print(os.listdir(os.path.dirname(mesh_path)))

        sim  = self.initialize_sim_scene(
            num_sensors, mesh_path, sim=sim, max_distance=max_distance
        )

        #create folder for examples
        write_dir = os.path.join(data_write_path, instance_name)
        os.makedirs(write_dir, exist_ok=True)

        """
        generate num_states unique configurations in this room
        with this AudioSensor count.
        """
        for i in tqdm(range(num_states)): 
            island_idx = self.get_largest_island_for_scene(sim)
            self.set_random_audio_source_positions(sim, num_sensors, island_idx)
            random_sources_indices = [
                random.randint(0, len(self.data_paths)-1) for _ in range(num_sensors)
            ]
            source_paths = [self.data_paths[i] for i in random_sources_indices]
            source_sequence, sr = self.create_audio_cue_sequence(
                source_paths, step_duration=step_duration, 
            )

            print(f"   -> Generating instnance {i} for scene at : {mesh_path}")
            data_instance = self.generate_stationary_data_instance(sim, source_sequence)
            print("    -> Generation done")


            #make instance write dir
            instance_write_dir = os.path.join(write_dir, f"instance_{i}")
            os.makedirs(instance_write_dir, exist_ok=True)

            write_path_audio = os.path.join(instance_write_dir, f"mix_audio.wav")
            write_path_counts = os.path.join(instance_write_dir, f"counts.npy")
            write_path_source = os.path.join(instance_write_dir, f"source")

            #write audio and source counts
            sf.write(write_path_audio, data_instance['mix_audio'].T, int(sr))
            np.save(write_path_counts, data_instance['source_counts'])

            if return_sources: 
                for i, src in enumerate(data_instance["sources"]): 
                    path = write_path_source + f"_{i}.wav"
                    sf.write(path, src.T, int(sr))

        return sim

    """
    @brief: generates all dynamic instance of audio for a scene
    """
    def generate_dynamic_instances_for_scene(self): 
        pass

    """
    build_dataset

    @breif: builds the whole train dataset
    @Note: Need to edit audio source writing so that audio 
    sources are same length as audio mixes for slicing. In essence
    audio sources need to be the target split of audio mixes

    bad_scenes: 
        8, 35, 51, 55, 78
    """
    def build_dataset(
        self, 
        num_states_per_scene: int, 
        max_min_sensors: tuple[int, int], 
        data_write_path: str, 
        return_sources: bool = True,
        step_duration: float = 0.2, 
        mesh_type: str = "glb",
        max_distance: float = 10.0
    ) -> None: 
        os.makedirs(data_write_path, exist_ok=True)

        lo, hi = max_min_sensors
        prog_bar = tqdm(
            enumerate(self.mesh_paths), 
            desc="Building Scene Data"
        )
        sim = None
        for i, path in prog_bar:
            if i <= 49: continue
            if i in [8, 25, 35, 51, 55, 78, 86, 96, 112]: continue 
            instance_name = f"scene_{i}"
            num_sensors = random.randint(lo, hi)
            sim = self.generate_stationary_instances_for_scene(
                path, num_sensors, num_states_per_scene, instance_name, 
                data_write_path, return_sources, step_duration, mesh_type,
                sim=sim, max_distance=max_distance
            )
