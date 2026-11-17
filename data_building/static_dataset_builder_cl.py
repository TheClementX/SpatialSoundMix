"""
static_dataset_builder_cl.py

Builds the static SpatialSoundMix dataset with habitat-sim.

habitat-sim ray traces a binaural room impulse response (RIR) from every audio source
to the listener (the agent). The sources hold no audio: a scene with n placed sources
gives n RIRs, and any n recordings can be spatialised with them.

For every HM3D scene, with a random number of sources:

    for each of num_states placements:
        1. place the listener and the sources at random and ray trace one RIR per source
        2. pick one recording per source, each from a different speaker, and give each
           a random start time in the mix
        3. convolve each recording with its RIR, raise speakers that would be too quiet
           (max_level_diff_db) and sum them into the binaural mix
        4. write the mix, the count of active speakers per sample and, optionally, one
           separation target per speaker (see SOURCE_TYPES)

Nothing moves, so each recording is convolved with its RIR once over its whole length.

Every scene runs in its own process (see build_dataset): a scene that crashes, hangs or
runs out of memory inside habitat-sim is skipped instead of stopping the build, and
scenes already written are skipped when the build is restarted.

Output, in data_write_path/scene_<i>/instance_<k>/:
    mix_audio.wav   (samples, 2) binaural mix, peak normalised to 1
    counts.npy      (2, samples) number of active speakers at each sample
    source_<j>.wav  separation target of speaker j, same length as the mix, (samples, 2)
                    if binaural_sources else (samples,): dry targets at their recorded
                    level, direct / full targets at the mix's scale (32-bit float, so a
                    peak slightly over 1 is not clipped in the file)
"""

import os

#cap the BLAS / OpenMP thread pools before numpy is imported. Scene processes inherit
#this through their environment; ray tracing threads are set with rlr_threads instead.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import multiprocessing as mp
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass

import habitat_sim
import numpy as np
import scipy.signal as ss
import soundfile as sf
from tqdm import tqdm

from material_fallbacks_cl import write_scene_material_file

#separation target written for each speaker:
#   "dry"    the clean recording, at the level it was recorded at
#   "direct" the recording convolved with the direct path of its RIR
#   "full"   the recording convolved with its full RIR (reverberant);
#            the full targets of a mix sum to the mix
#each is written binaural (2 channels) or mono (1 channel), see binaural_sources
SOURCE_TYPES = ("dry", "direct", "full")

#what a mono direct / full target is: the average of both ears, or one ear, by its
#channel index in habitat-sim's binaural RIR
MONO_CHANNELS = {"average": None, "left": 0, "right": 1}

#exit code a scene process uses to stop the whole build (see AudioSourcePositionError)
_FATAL_EXIT_CODE = 3


class AudioSourcePositionError(RuntimeError):
    """habitat-sim does not simulate audio sources where they are put. This stops the
    whole build, because every scene would be simulated wrongly."""


# --------------------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------------------

@dataclass
class AudioCue:
    """One recording in a mix: mono audio at the simulation rate, where it is active,
    and the sample at which it starts in the mix."""
    path: str
    audio: np.ndarray
    active: np.ndarray
    start: int = 0

    @classmethod
    def load(cls, path: str, sample_rate: float, threshold: float = 0.01) -> "AudioCue":
        audio, file_rate = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        #resample to the simulation rate (EARS is recorded at 48 kHz); without this
        #48 kHz speech plays back ~9% slow and pitched down
        if file_rate != sample_rate:
            g = np.gcd(int(sample_rate), int(file_rate))
            audio = ss.resample_poly(audio, int(sample_rate) // g, int(file_rate) // g)
        #active where the Hilbert envelope is above threshold
        #TODO: mathematically check the validity of this activity counting approach
        active = np.abs(ss.hilbert(audio)) > threshold
        return cls(path, audio, active)

    @property
    def num_samples(self) -> int:
        return len(self.audio)


def convolve(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Full convolution of mono audio with a (channels, taps) RIR -> (channels, samples)."""
    return ss.fftconvolve(audio[np.newaxis, :], rir, axes=1)


def place(signal: np.ndarray, start: int, num_samples: int) -> np.ndarray:
    """signal (..., n) delayed by start samples and cut or zero padded to num_samples."""
    out = np.zeros(signal.shape[:-1] + (num_samples,))
    end = min(start + signal.shape[-1], num_samples)
    out[..., start:end] = signal[..., :end - start]
    return out


# --------------------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------------------

class SpatialSoundMixDatasetBuilder:
    #binaural listener
    NUM_CHANNELS = 2
    #minimum distance between a speaker and the listener (m)
    MIN_SOURCE_DISTANCE = 0.5
    #attempts at a valid placement / audible configuration before a scene is skipped
    MAX_PLACEMENT_TRIES = 20
    #placements whose weakest RIR is further below the strongest (dB) are resampled:
    #those speakers are sealed off from the listener or placed outside the mesh
    MAX_RIR_RANGE_DB = 30.0
    #navmesh islands smaller than this (m^2) mark the scene as invalid
    MIN_ISLAND_AREA = 5.0
    #silence around the speech in a mix (s), split between its start and end
    MIX_PADDING = 0.5
    #direct path for source_type "direct": the RIR from DIRECT_PATH_PRE before the first
    #arrival to DIRECT_PATH_POST after it (s), with tapered edges. The first arrival in a
    #channel is its first sample within DIRECT_PATH_THRESHOLD of that channel's peak.
    DIRECT_PATH_THRESHOLD = 0.2
    DIRECT_PATH_PRE = 0.001
    DIRECT_PATH_POST = 0.0025

    def __init__(
        self,
        data_dirs: list[str],
        mesh_dirs: list[str],
        audio_absorption_file: str,
        dataset_config_file: str,
        sample_rate: float = 44100.0,
        seed: int = 42,
        source_type: str = "dry",
        binaural_sources: bool = True,
        mono_channel: str = "left",
        max_level_diff_db: float = 6.0,
        rlr_threads: int = 1,
        instance_timeout: float = 1800.0,
        filter_by_semantic_scene: bool = True,
        verify_source_positions: bool = False
    ) -> None:
        """
        @param:
            data_dirs: one folder of recordings per speaker
            mesh_dirs: folders holding one HM3D scene folder per scene
            audio_absorption_file: RLR material file (fallbacks are added per scene)
            dataset_config_file: HM3D scene dataset config
            source_type: separation target written per speaker, one of SOURCE_TYPES
            binaural_sources: write targets with two channels (left / right ear) rather
                than one (see source_target)
            mono_channel: with binaural_sources False, the channel a direct / full
                target keeps, one of MONO_CHANNELS
            max_level_diff_db: no speaker ends up more than this far below the loudest
                speaker at the listener (see audibility_gains)
            rlr_threads: CPU threads habitat-sim's ray tracer may use
            instance_timeout: seconds allowed per instance before a scene process is
                killed and the scene skipped
            filter_by_semantic_scene: only use scenes with a .semantic.glb
            verify_source_positions: check habitat-sim's source placement once before
                the first scene (see verify_audio_source_positions)
        """
        if source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {SOURCE_TYPES}, not {source_type!r}")
        if mono_channel not in MONO_CHANNELS:
            raise ValueError(
                f"mono_channel must be one of {tuple(MONO_CHANNELS)}, not {mono_channel!r}"
            )

        self.audio_absorption_file = audio_absorption_file
        self.dataset_config_file = dataset_config_file
        self.sample_rate = sample_rate
        self.seed = seed
        self.source_type = source_type
        self.binaural_sources = binaural_sources
        self.mono_channel = mono_channel
        self.max_level_diff_db = max_level_diff_db
        self.rlr_threads = rlr_threads
        self.instance_timeout = instance_timeout
        #set once a scene has been built with verification on
        self.source_positions_verified = not verify_source_positions

        self.speaker_paths = self.find_speakers(data_dirs)
        self.mesh_paths = self.find_scenes(mesh_dirs, filter_by_semantic_scene)

    # ----------------------------------------------------------------------------------
    # dataset files
    # ----------------------------------------------------------------------------------

    @staticmethod
    def find_speakers(data_dirs: list[str]) -> dict[str, list[str]]:
        """Recordings grouped by speaker folder, so a mix never has one voice twice."""
        speakers = {}
        for d in data_dirs:
            if not os.path.isdir(d):
                continue
            files = sorted(
                os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".wav", ".flac"))
            )
            if files:
                speakers[d] = files
        return speakers

    @staticmethod
    def find_scenes(mesh_dirs: list[str], require_semantics: bool) -> list[str]:
        """Scene folders, optionally only those with a semantic mesh."""
        scenes = []
        for d in mesh_dirs:
            for name in sorted(os.listdir(d)):
                path = os.path.join(d, name)
                semantic_mesh = os.path.join(path, name.split("-")[-1] + ".semantic.glb")
                if os.path.isdir(path) and (not require_semantics or os.path.isfile(semantic_mesh)):
                    scenes.append(path)
        return scenes

    @staticmethod
    def find_mesh(mesh_dir: str) -> str:
        """The .basis.glb mesh of a scene folder, checked to have a navmesh."""
        meshes = sorted(p for p in os.listdir(mesh_dir) if p.endswith(".basis.glb"))
        if not meshes:
            raise ValueError(f"no .basis.glb mesh in {mesh_dir}")
        mesh_path = os.path.join(mesh_dir, meshes[0])
        if not os.path.isfile(mesh_path[:-4] + ".navmesh"):
            raise ValueError(f"no navmesh next to {mesh_path}")
        return mesh_path

    # ----------------------------------------------------------------------------------
    # simulator
    # ----------------------------------------------------------------------------------

    def initialize_sim_scene(
        self,
        num_sources: int,
        mesh_path: str,
        seed: int
    ) -> tuple[habitat_sim.Simulator, int]:
        """Load a scene with num_sources binaural audio sensors.
        @return: the simulator and the index of the navmesh island to place things on
        """
        print(f"loading scene with {num_sources} sources at: {mesh_path}")
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_dataset_config_file = self.dataset_config_file
        #scene handle, not the full file path
        backend_cfg.scene_id = mesh_path.split("hm3d/")[-1]
        backend_cfg.load_semantic_mesh = True
        #without this habitat-sim misreads the semantic.glb as geometry
        backend_cfg.use_semantic_textures = False
        backend_cfg.enable_physics = True
        backend_cfg.random_seed = seed

        acoustics_cfg = habitat_sim.RLRAudioPropagationConfiguration()
        acoustics_cfg.enableMaterials = True
        acoustics_cfg.temporalCoherence = False
        acoustics_cfg.threadCount = self.rlr_threads
        acoustics_cfg.sampleRate = int(self.sample_rate)

        sensor_specs = []
        for i in range(num_sources):
            spec = habitat_sim.AudioSensorSpec()
            spec.uuid = f"audio_source_{i}"
            spec.acousticsConfig = acoustics_cfg
            spec.channelLayout.channelType = (
                habitat_sim.RLRAudioPropagationChannelLayoutType.Binaural
            )
            sensor_specs.append(spec)

        agent_cfg = habitat_sim.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
        sim.seed(seed)

        if len(sim.semantic_scene.objects) == 0:
            raise ValueError("semantics not loaded")

        if not sim.pathfinder.load_nav_mesh(mesh_path[:-4] + ".navmesh"):
            raise ValueError("attempted loading an invalid navmesh")
        island = self.largest_island(sim)
        island_area = sim.pathfinder.island_area(island)
        if island_area < self.MIN_ISLAND_AREA:
            raise ValueError(f"largest navmesh island is only {island_area:.1f} m^2")

        #a per-scene copy of the material file with fallbacks for unlabelled categories
        material_file = self.write_scene_materials(sim)
        for sensor in sim.get_agent(0)._sensors.values():
            sensor.setAudioMaterialsJSON(material_file)

        if not self.source_positions_verified:
            self.verify_audio_source_positions(sim, island)
        print("scene initialized")
        return sim, island

    @staticmethod
    def largest_island(sim: habitat_sim.Simulator) -> int:
        pf = sim.pathfinder
        if pf.num_islands == 0:
            raise ValueError("navmesh has no navigable islands")
        return int(np.argmax([pf.island_area(i) for i in range(pf.num_islands)]))

    def scene_material_path(self) -> str:
        """Where this process keeps the scene's augmented material file."""
        return os.path.join(tempfile.gettempdir(), f"spatialsoundmix_materials_{os.getpid()}.json")

    def write_scene_materials(self, sim: habitat_sim.Simulator) -> str:
        """Write this scene's material file (see material_fallbacks_cl).
        @return: its path
        """
        categories = {
            obj.category.name() for obj in sim.semantic_scene.objects
            if obj is not None and obj.category is not None
        }
        path = self.scene_material_path()
        num_fallback, num_default, num_unstable = write_scene_material_file(
            self.audio_absorption_file, categories, path
        )
        print(
            f"materials: {len(categories)} categories, {num_fallback} without a label match, "
            f"{num_fallback - num_default} given a fallback material, "
            f"{num_unstable} not resolving as intended"
        )
        return path

    def verify_audio_source_positions(
        self,
        sim: habitat_sim.Simulator,
        island: int,
        far_offset: float = 200.0
    ) -> None:
        """Make sure habitat-sim simulates sources where they are placed.

        habitat-sim >= 0.3.1 hands the ray tracer lastSourcePos_[1], [2], [3] in
        AudioSensor::runSimulation (src/esp/sensor/AudioSensor.cpp), i.e. (y, z, <garbage>)
        instead of (x, y, z), which puts most speakers outside the house where they are
        barely heard. A source 0.3 m above the listener is compared with the same source
        moved 200 m along x: a correct build hears the far one tens of dB quieter, the
        broken build ignores x and returns the same RIR.
        @note: only sensor 0 is run, through the same calls habitat-sim's own sensor
        wrapper makes.
        """
        agent = sim.get_agent(0)
        sensor = agent._sensors["audio_source_0"]

        def rir_energy(position):
            sensor.setAudioSourceTransform(np.asarray(position, dtype=np.float32))
            rot = agent.get_state().rotation
            sensor.setAudioListenerTransform(
                sensor.node.absolute_translation, np.array([rot.w, rot.x, rot.y, rot.z])
            )
            sensor.runSimulation(sim)
            rir = np.asarray(sensor.getIR(), dtype=np.float64)
            return float(np.sum(rir ** 2)) if rir.size and np.all(np.isfinite(rir)) else 0.0

        for _ in range(self.MAX_PLACEMENT_TRIES):
            agent_state = habitat_sim.AgentState()
            agent_state.position = sim.pathfinder.get_random_navigable_point(island_index=island)
            if not np.all(np.isfinite(agent_state.position)):
                continue
            agent.set_state(agent_state)

            #the probe source needs head room above the listener
            listener_pos = np.array(sensor.node.absolute_translation, dtype=np.float32)
            hits = sim.cast_ray(habitat_sim.geo.Ray(listener_pos, np.array([0.0, 1.0, 0.0]))).hits
            if len(hits) > 0 and hits[0].ray_distance < 0.5:
                continue

            near = listener_pos + np.array([0.0, 0.3, 0.0], dtype=np.float32)
            near_energy = rir_energy(near)
            if near_energy <= 0.0:
                continue
            far_energy = rir_energy(near + np.array([far_offset, 0.0, 0.0], dtype=np.float32))
            if far_energy < 0.1 * near_energy:
                return
            break

        raise AudioSourcePositionError(
            "habitat-sim is not simulating audio sources where they are placed: "
            "moving a source along x does not change its RIR, or a source 0.3 m from the "
            "listener is silent. In habitat-sim >= 0.3.1, AudioSensor::runSimulation in "
            "src/esp/sensor/AudioSensor.cpp calls AddSource with lastSourcePos_[1], "
            "lastSourcePos_[2], lastSourcePos_[3]. Change these to lastSourcePos_[0], "
            "lastSourcePos_[1], lastSourcePos_[2], rebuild habitat-sim with audio and run "
            "again. To build anyway, pass verify_source_positions=False."
        )

    # ----------------------------------------------------------------------------------
    # placement
    # ----------------------------------------------------------------------------------

    def place_sources(
        self,
        sim: habitat_sim.Simulator,
        num_sources: int,
        island: int,
        max_distance: float = float("inf")
    ) -> list[np.ndarray]:
        """Put the listener and the sources at random navigable points and ray trace.
        @return: one (channels, taps) RIR per source, or None if the placement is
        invalid (see rirs_valid)
        """
        agent = sim.get_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_pos = sim.pathfinder.get_random_navigable_point(island_index=island)
        if not np.all(np.isfinite(agent_pos)):
            return None
        agent_state.position = agent_pos
        agent.set_state(agent_state)
        listener_pos = np.array(
            agent._sensors["audio_source_0"].node.absolute_translation, dtype=np.float32
        )

        for i in range(num_sources):
            for _ in range(self.MAX_PLACEMENT_TRIES):
                floor_pos = sim.pathfinder.get_random_navigable_point_near(
                    circle_center=agent_pos, radius=max_distance, max_tries=400,
                    island_index=island
                )
                path = habitat_sim.ShortestPath()
                path.requested_start = agent_pos
                path.requested_end = floor_pos
                if not sim.pathfinder.find_path(path):
                    floor_pos = sim.pathfinder.get_random_navigable_point(island_index=island)
                if not np.all(np.isfinite(floor_pos)):
                    continue

                source_pos = self.random_elevation(sim, floor_pos)
                #keep speakers out of the listener's head
                if np.linalg.norm(source_pos - listener_pos) >= self.MIN_SOURCE_DISTANCE:
                    break
            else:
                return None
            agent._sensors[f"audio_source_{i}"].setAudioSourceTransform(source_pos)

        obs = sim.get_sensor_observations()
        rirs = [
            np.asarray(obs.get(f"audio_source_{i}", []), dtype=np.float64)
            for i in range(num_sources)
        ]
        if not self.rirs_valid(rirs):
            return None
        return [rir[:self.NUM_CHANNELS] for rir in rirs]

    def rirs_valid(self, rirs: list[np.ndarray]) -> bool:
        """Every source produced a usable RIR: present, finite, non-silent, and no more
        than MAX_RIR_RANGE_DB below the strongest one. Quieter RIRs come from speakers
        sealed off from the listener or placed outside the mesh."""
        for rir in rirs:
            if (
                rir.ndim != 2 or rir.shape[0] < self.NUM_CHANNELS or rir.shape[1] == 0
                or not np.all(np.isfinite(rir))
            ):
                return False
        energies = [np.sum(rir ** 2) for rir in rirs]
        if min(energies) <= 0.0:
            return False
        return 10.0 * np.log10(max(energies) / min(energies)) <= self.MAX_RIR_RANGE_DB

    @staticmethod
    def random_elevation(
        sim: habitat_sim.Simulator,
        point: np.ndarray,
        default_height: float = 1.6,
        min_height: float = 0.05,
        ceiling_offset: float = 0.05
    ) -> np.ndarray:
        """Lift a floor point to a random speaker height below the ceiling.
        @param:
            default_height: the maximum height where no ceiling is hit
            min_height: the minimum height
            ceiling_offset: how far below the ceiling a point stays
        """
        point = np.asarray(point, dtype=np.float32) + np.array([0.0, 0.05, 0.0])
        hits = sim.cast_ray(habitat_sim.geo.Ray(point, np.array([0.0, 1.0, 0.0]))).hits
        if len(hits) > 0:
            max_height = max(min_height, hits[0].ray_distance - ceiling_offset)
        else:
            max_height = default_height

        new_point = point.copy()
        new_point[1] += random.uniform(min_height, max_height)
        return new_point

    # ----------------------------------------------------------------------------------
    # mixing
    # ----------------------------------------------------------------------------------

    def sample_cues(self, num_sources: int) -> tuple[list[AudioCue], int]:
        """One recording per source, each from a different speaker, at random start times.
        @return: the cues and the length of the mix in samples
        """
        if num_sources > len(self.speaker_paths):
            raise ValueError(
                f"{num_sources} sources requested but only {len(self.speaker_paths)} speakers found"
            )
        speakers = random.sample(sorted(self.speaker_paths), num_sources)
        cues = [
            AudioCue.load(random.choice(self.speaker_paths[s]), self.sample_rate)
            for s in speakers
        ]

        padding = int(self.MIX_PADDING * self.sample_rate)
        num_samples = max(c.num_samples for c in cues) + padding
        for c in cues:
            c.start = random.randint(padding // 2, num_samples - padding // 2 - c.num_samples)
        return cues, num_samples

    def render_instance(
        self,
        cues: list[AudioCue],
        num_samples: int,
        rirs: list[np.ndarray],
        with_sources: bool = True
    ) -> dict:
        """Spatialise the cues with their RIRs into one binaural mix.

        Each recording is convolved with its RIR once, over its whole length, and added at
        its start sample; reverb tails running past the end of the mix are cut.
        @return: {"mix_audio": (2, N), "source_counts": (2, N), "sources": [...]}, or
        None if some speaker cannot be made audible
        """
        wet = [convolve(c.audio, rir) for c, rir in zip(cues, rirs)]
        gains = self.audibility_gains(cues, wet)
        if gains is None:
            return None

        mix = sum(g * place(w, c.start, num_samples) for c, w, g in zip(cues, wet, gains))
        peak = np.max(np.abs(mix))
        if peak <= 0.0:
            return None

        counts = sum(place(c.active, c.start, num_samples) for c in cues)
        instance = {
            "mix_audio": mix / peak,
            "source_counts": np.repeat(counts[np.newaxis], self.NUM_CHANNELS, axis=0),
        }
        if with_sources:
            instance["sources"] = [
                place(self.source_target(c, rir, w, g / peak), c.start, num_samples)
                for c, rir, w, g in zip(cues, rirs, wet, gains)
            ]
        return instance

    def audibility_gains(self, cues: list[AudioCue], wet: list[np.ndarray]) -> np.ndarray:
        """Guarantee that every speaker is audible in the mix.

        A speaker's level at the listener is the RMS of its wet signal over the samples
        where it is active, summed over both ears. Any speaker more than
        max_level_diff_db below the loudest is raised to exactly that bound; the others
        keep their natural distance / occlusion level. The gain is the same for both
        ears, so interaural and direct-to-reverberant cues are untouched.
        @param:
            wet: each cue convolved with its RIR, (channels, samples)
        @return: one linear gain per cue, or None if a speaker never reaches the listener
        """
        levels = []
        for c, w in zip(cues, wet):
            active = c.active if c.active.any() else np.ones_like(c.active)
            energy = np.sum(np.mean(w[:, :c.num_samples][:, active] ** 2, axis=1))
            levels.append(10.0 * np.log10(energy) if energy > 0 else -np.inf)

        levels = np.asarray(levels)
        if not np.all(np.isfinite(levels)):
            return None
        boost_db = np.maximum(0.0, levels.max() - self.max_level_diff_db - levels)
        return 10.0 ** (boost_db / 20.0)

    def source_target(
        self,
        cue: AudioCue,
        rir: np.ndarray,
        wet: np.ndarray,
        mix_gain: float
    ) -> np.ndarray:
        """The separation target of one speaker, before its placement in the mix.

        Direct and full targets get mix_gain, the speaker's gain and normalisation in
        the mix, so they are at the mix's scale and the full targets sum to the mix.
        A dry target keeps the level it was recorded at: it skips the RIR's attenuation,
        so at the mix's scale it would land far above full scale and clip.
        @return: (2, samples) if binaural_sources, else (samples,)
        """
        if self.source_type == "dry":
            target = cue.audio
        elif self.source_type == "direct":
            target = mix_gain * convolve(cue.audio, self.direct_path_rir(rir))
        else:
            target = mix_gain * wet

        if self.binaural_sources:
            #a dry recording has no ear differences: the same signal in both channels
            if target.ndim == 1:
                target = np.repeat(target[np.newaxis], self.NUM_CHANNELS, axis=0)
        elif target.ndim == 2:
            #mono: one ear, or the average of both
            if self.mono_channel == "average":
                target = target.mean(axis=0)
            else:
                target = target[MONO_CHANNELS[self.mono_channel]]
        return target

    def direct_path_rir(self, rir: np.ndarray) -> np.ndarray:
        """The direct path part of a (channels, taps) RIR, at its original delay.

        One window covers every channel, from DIRECT_PATH_PRE before the earliest
        first arrival to DIRECT_PATH_POST after the latest, so the interaural time
        and level differences of the direct sound are kept.
        """
        magnitude = np.abs(rir)
        onsets = np.argmax(magnitude >= self.DIRECT_PATH_THRESHOLD * magnitude.max(axis=1, keepdims=True), axis=1)
        start = max(0, onsets.min() - int(self.DIRECT_PATH_PRE * self.sample_rate))
        end = min(rir.shape[1], onsets.max() + int(self.DIRECT_PATH_POST * self.sample_rate) + 1)

        direct = np.zeros((rir.shape[0], end))
        direct[:, start:end] = rir[:, start:end] * ss.windows.tukey(end - start, alpha=0.5)
        return direct

    # ----------------------------------------------------------------------------------
    # generation
    # ----------------------------------------------------------------------------------

    def generate_instance(
        self,
        sim: habitat_sim.Simulator,
        island: int,
        num_sources: int,
        max_distance: float,
        with_sources: bool
    ) -> dict:
        """One placement and mix, retried until every speaker is audible."""
        for _ in range(self.MAX_PLACEMENT_TRIES):
            rirs = self.place_sources(sim, num_sources, island, max_distance)
            if rirs is None:
                continue
            cues, num_samples = self.sample_cues(num_sources)
            instance = self.render_instance(cues, num_samples, rirs, with_sources)
            if instance is not None:
                return instance
        raise RuntimeError(
            f"no configuration with every speaker audible after {self.MAX_PLACEMENT_TRIES} tries"
        )

    def write_instance(self, instance_dir: str, instance: dict) -> None:
        """Write one instance. It goes to a hidden temporary folder that is renamed once
        complete, so an interrupted run never leaves half an instance."""
        parent, name = os.path.split(instance_dir)
        tmp_dir = os.path.join(parent, f".{name}.tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)

        sr = int(self.sample_rate)
        sf.write(os.path.join(tmp_dir, "mix_audio.wav"), instance["mix_audio"].T, sr)
        np.save(os.path.join(tmp_dir, "counts.npy"), instance["source_counts"])
        for j, src in enumerate(instance.get("sources", [])):
            sf.write(os.path.join(tmp_dir, f"source_{j}.wav"), src.T, sr, subtype="FLOAT")

        shutil.rmtree(instance_dir, ignore_errors=True)
        os.rename(tmp_dir, instance_dir)

    def generate_scene(
        self,
        mesh_dir: str,
        num_sources: int,
        num_states: int,
        write_dir: str,
        return_sources: bool,
        max_distance: float,
        seed: int
    ) -> None:
        """Generate and write num_states placements of num_sources speakers in a scene."""
        mesh_path = self.find_mesh(mesh_dir)
        sim, island = self.initialize_sim_scene(num_sources, mesh_path, seed)
        os.makedirs(write_dir, exist_ok=True)

        for k in tqdm(range(num_states), desc="Instances", leave=False):
            instance = self.generate_instance(sim, island, num_sources, max_distance, return_sources)
            self.write_instance(os.path.join(write_dir, f"instance_{k}"), instance)
        sim.close()

    # ----------------------------------------------------------------------------------
    # build
    # ----------------------------------------------------------------------------------

    @staticmethod
    def scene_is_complete(write_dir: str, num_states: int) -> bool:
        """True when every instance of a scene has been written."""
        return all(
            os.path.isfile(os.path.join(write_dir, f"instance_{k}", name))
            for k in range(num_states) for name in ("mix_audio.wav", "counts.npy")
        )

    def check_build_config(self, data_write_path: str, return_sources: bool, max_distance: float) -> None:
        """Refuse to resume a build made with other settings: finished scenes are skipped,
        so the dataset would silently mix both."""
        config = {
            "source_type": self.source_type,
            "binaural_sources": self.binaural_sources,
            "mono_channel": self.mono_channel,
            "return_sources": return_sources,
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "max_level_diff_db": self.max_level_diff_db,
            "max_distance": max_distance,
        }
        path = os.path.join(data_write_path, "build_config.json")
        if os.path.isfile(path):
            with open(path) as f:
                previous = json.load(f)
            if previous != config:
                raise ValueError(
                    f"{data_write_path} was built with {previous}, not {config}. "
                    "Use another write path, or delete the old build."
                )
        else:
            with open(path, "w") as f:
                json.dump(config, f, indent=2)

    def _scene_process(
        self,
        scene_idx: int,
        mesh_dir: str,
        num_sources: int,
        num_states: int,
        write_dir: str,
        return_sources: bool,
        max_distance: float
    ) -> None:
        """Body of the process build_dataset starts for one scene. The process owns the
        simulator for its scene only, so memory and threads cannot pile up across scenes
        and a crash inside habitat-sim ends only this scene."""
        #seeded per scene, so a scene comes out the same whatever ran before it
        random.seed(self.seed + scene_idx)
        #start clean: drops partial output of an interrupted earlier run
        shutil.rmtree(write_dir, ignore_errors=True)
        try:
            self.generate_scene(
                mesh_dir, num_sources, num_states, write_dir, return_sources, max_distance,
                seed=self.seed + scene_idx
            )
        except AudioSourcePositionError as e:
            print(f"\n{e}", file=sys.stderr, flush=True)
            sys.exit(_FATAL_EXIT_CODE)
        finally:
            if os.path.exists(self.scene_material_path()):
                os.remove(self.scene_material_path())

    def build_dataset(
        self,
        num_states_per_scene: int,
        num_sources_range: tuple[int, int],
        data_write_path: str,
        return_sources: bool = True,
        max_distance: float = 10.0
    ) -> None:
        """Build the whole static dataset.

        Each scene runs in its own process, one at a time, with a random number of
        sources in num_sources_range (min, max). A scene that raises, segfaults, is
        killed for memory or runs past its timeout is skipped and its partial output
        removed. Scenes already written are skipped, so an interrupted build resumes
        where it stopped.
        """
        os.makedirs(data_write_path, exist_ok=True)
        self.check_build_config(data_write_path, return_sources, max_distance)

        lo, hi = num_sources_range
        #spawn, not fork: a fresh interpreter per scene, no inherited GL / thread state
        ctx = mp.get_context("spawn")
        skipped = {}
        prog_bar = tqdm(
            enumerate(self.mesh_paths), total=len(self.mesh_paths), desc="Building Scene Data"
        )
        for i, mesh_dir in prog_bar:
            write_dir = os.path.join(data_write_path, f"scene_{i}")
            if self.scene_is_complete(write_dir, num_states_per_scene):
                continue
            num_sources = random.Random(self.seed + i).randint(lo, hi)

            proc = ctx.Process(
                target=self._scene_process,
                args=(i, mesh_dir, num_sources, num_states_per_scene, write_dir,
                      return_sources, max_distance)
            )
            proc.start()
            try:
                proc.join(self.instance_timeout * num_states_per_scene)
                timed_out = proc.is_alive()
            finally:
                if proc.is_alive():
                    proc.kill()
                proc.join()

            #success is judged by what is on disk, so a crash while the scene
            #process tears habitat-sim down does not discard finished work
            if self.scene_is_complete(write_dir, num_states_per_scene):
                self.source_positions_verified = True
                continue

            shutil.rmtree(write_dir, ignore_errors=True)
            if proc.exitcode == _FATAL_EXIT_CODE:
                raise AudioSourcePositionError(
                    "stopping: habitat-sim does not place audio sources correctly (see above)"
                )
            skipped[i] = "timed out" if timed_out else f"exit code {proc.exitcode}"
            prog_bar.set_postfix(skipped=len(skipped))

        if skipped:
            print(f"{len(skipped)} scene(s) skipped:")
            for i, reason in skipped.items():
                print(f"   scene_{i} ({self.mesh_paths[i]}): {reason}")
