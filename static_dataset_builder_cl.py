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

Robustness: every scene is simulated in its own process (see build_dataset), so a scene
that crashes, hangs or exhausts memory inside habitat-sim is skipped instead of taking
the whole build down, and finished scenes are skipped when the build is restarted.
"""

import os

#cap the BLAS / OpenMP thread pools before numpy is imported. Scene processes inherit
#this through their environment; ray tracing threads are set with rlr_threads instead.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import copy
import difflib
import json
import multiprocessing as mp
import re
import shutil
import sys
import tempfile
import habitat_sim
import numpy as np
import soundfile as sf
import scipy.signal as ss 
import random
from tqdm import tqdm 
import gc

#exit code a scene process uses to stop the whole build (see AudioSourcePositionError)
_FATAL_EXIT_CODE = 3

"""
@brief: Fallback acoustic materials for HM3D semantic categories.

RLR gives each mesh category the material with the most labels that are substrings
of the lower-cased category name. The mp3d material file only labels mpcat40-style
names, so about two thirds of the raw HM3D vocabulary ("couch", "lamp", "ceilling",
"kitchen island", ...) matches nothing and is simulated with the engine default.

Each entry maps a target to keywords found in HM3D names. A target is either a label
already in the material file, so the category inherits that file's own choice
("couch" -> "sofa" -> Curtain), or a material name for classes the file never labels
(pictures, paper, outdoor ground, small objects). Keywords are matched as whole words
(plurals and close typos included) with the head noun first; multi-word keywords are
matched as phrases.
"""
ACOUSTIC_FALLBACKS = {
    #structure
    "ceiling": "roof, soffit",
    "wall": "baseboard, skirting, molding, moulding, cornice, paneling, panelling, niche, "
            "relief, partition, parapet, column, pillar, arch, archway, ledge",
    "floor": "landing, platform, stage",
    "window": "skylight, stained glass",
    "door": "gate, hatch, doorway, entrance",
    "stairs": "step, ladder, riser",
    "railing": "balustrade, banister, bannister, handhold, rail, fence, grab bar, bath bar",
    "beam": "rafter, balk, joist, truss",
    "fireplace": "chimney, hearth, mantel, mantle, firebox, wood burner",
    #outdoor ground and openings
    "Concrete, Rough": "pavement, paving, patio, terrace, driveway, sidewalk, curb, "
                       "balcony, porch, deck, courtyard",
    "Grass": "lawn, grass, turf",
    "Gravel": "gravel",
    "Soil": "dirt, soil, ground, mud",
    "Water": "pool, pond, fountain",
    "Sound Proof": "sky",
    "plant": "tree, flower, bouquet, bush, shrub, hedge, vegetation, foliage, leaf, leaves, "
             "wreath, twig, branch, bonsai",
    #soft furnishings and textiles
    "cushion": "pillow, pouf, pouffe, poufe, ottoman, foam",
    "sofa": "couch, loveseat, futon, chaise",
    "bed": "mattress, crib, cradle, cot, bunk, headboard, hammock",
    "clothes": "coat, jacket, shirt, blouse, dress, robe, bathrobe, laundry, sock, hat, "
               "glove, shoe, boot, sandal, slipper, apron",
    "curtain": "drape, drapery, tapestry, banner, flag, canopy, tent, teepee, linen, sheet, "
               "duvet, quilt, doily, napkin, rag, handkerchief, tissue, fabric, textile, "
               "wool, fur, plush, stuffed, teddy, doll, puppet, lamp shade, lampshade",
    "Carpet": "carpet",
    "blinds": "blind, blinder, shade, shutter",
    #furniture
    "chest_of_drawers": "dresser, drawer, bureau, chest, commode",
    "cabinet": "closet, cupboard, pantry, hutch, sideboard, credenza, vitrine",
    "shelf": "shelve, bookcase, rack",
    "table": "stand, vanity, workstation",
    "counter": "island, worktop, bar, buffet",
    "seating": "bench, seat, lounger, recliner, backrest",
    "stool": "footrest, foot rest",
    "board_panel": "board, panel, plank, whiteboard, blackboard, noticeboard, corkboard, "
                   "dartboard",
    "furniture": "storage, unit, furnishing",
    "wood, Thick": "wood, wooden, timber, log, firewood, piano, sauna, stick",
    "wood, Thin": "picture, photo, photograph, photography, poster, painting, art, artwork, "
                  "canvas, portrait, image, drawing, frame, map, calendar, diploma, certificate, "
                  "sign, advertisement, menu, book, magazine, newspaper, paper, album, binder, "
                  "folder, notebook, document, file, brochure, leaflet, letter, card, cardboard, "
                  "box, crate, package, carton, basket, guitar, violin, ukulele",
    #hard, reflective
    "lighting": "lamp, light, chandelier, sconce, lantern, bulb, fixture",
    "tv_monitor": "tv, television, monitor, display, screen, laptop, computer, projector",
    "Glass": "glass, glazing, pane, bottle, jar, aquarium, fish tank",
    "appliances": "appliance, fridge, refrigerator, freezer, oven, stove, cooker, hob, stovetop, "
                  "dishwasher, washer, dryer, kettle, toaster, boiler, furnace, heater, radiator, "
                  "conditioner, dehumidifier, humidifier, purifier, fan, vent, ventilation, "
                  "ventilator, duct, hood, extractor, machine, maker, processor, compressor, "
                  "printer, copier, scanner, shredder, vacuum, mower, calculator, phone, "
                  "telephone, radio, stereo, player, soundbar, sound bar, speaker, subwoofer, "
                  "amplifier, amp, router, modem, icebox, electric, electrical, electronics, "
                  "device",
    "gym_equipment": "gym, exercise, fitness, workout, trainer, dumbbell, dumb bell, barbell, "
                     "weight, treadmill, trampoline",
    "Steel": "metal, steel, iron, aluminium, aluminum, chrome, brass, copper, pipe, piping, "
             "faucet, tap, valve, handle, knob, hook, rod, pole, chain, tank, can, tin, safe, "
             "locker, mailbox, bucket, pan, pot, cart, trolley, wheelbarrow, bicycle, bike, "
             "motorcycle, car, wheel, grate, grill, barbecue, extinguisher, trumpet, trombone, "
             "saxophone, clarinet, drum",
    "toilet": "urinal, bidet, potty, chamber pot",
    "sink": "basin, washbasin",
    "bathtub": "tub, jacuzzi, bath",
    "Tile, Ceramic": "ceramic, porcelain, tile, tiling, plate, dish, bowl, cup, mug, teapot, jug, "
                     "pitcher, vase, flowerpot, flower pot, platter, saucer, dinnerware, "
                     "backsplash, figurine",
    "Concrete": "stone, marble, granite, concrete, cement, stonework, rock, statue, sculpture, "
                "bust, slab",
    "Brick": "brick",
    "Acoustic Tile": "acoustic",
}

#words naming what something is made of win over the noun ("stone bench", "metal rack")
_MATERIAL_WORDS = {
    "glass", "metal", "steel", "iron", "aluminium", "aluminum", "chrome", "brass", "copper",
    "wood", "wooden", "timber", "stone", "marble", "granite", "concrete", "cement", "brick",
    "ceramic", "porcelain", "tile", "fabric", "textile", "wool", "fur", "foam", "acoustic",
}
_FALLBACK_KEYWORDS = {
    keyword.strip(): target
    for target, keywords in ACOUSTIC_FALLBACKS.items()
    for keyword in keywords.split(",")
}
#real words in the HM3D vocabulary that sit one typo away from a label or keyword
_NOT_TYPOS = {
    "shell", "paint", "tower", "plane", "self", "closing", "star", "wind", "brush",
    "marker", "title", "trailer",
}
#"box of tissues" / "basket /w books" / "flowers in vase": the head comes before these
_PHRASE_SPLIT = re.compile(r"\b(?:of|with|w|in|and|or|for|on)\b|[/,(]")


def _label_hits(category: str, materials: list[dict]) -> list[int]:
    category = category.lower()
    return [
        sum(1 for label in m.get("labels", []) if label and label.lower() in category)
        for m in materials
    ]


"""
@brief: mirror RLR's material lookup for one mesh category: the material whose labels
occur most often as substrings of the lower-cased name (first one wins a tie).
@return: the material index, or None when RLR would use its built-in default
"""
def rlr_material_index(category: str, materials: list[dict]):
    hits = _label_hits(category, materials)
    best = max(hits, default=0)
    return hits.index(best) if best > 0 else None


def _is_strict_winner(category: str, materials: list[dict], index: int) -> bool:
    hits = _label_hits(category, materials)
    return hits[index] > max((h for i, h in enumerate(hits) if i != index), default=0)


"""
@brief: choose a material for a category that none of the material file's labels match,
trying in order: spacing variants of a label ("chest of drawers" -> chest_of_drawers),
multi-word keywords, material words, nouns head first ("picture frame" -> frame,
"box of tissues" -> box), typo-tolerant matches against keywords and labels
("ceilling" -> ceiling), and finally the file's default material.
@param:
    category: the HM3D category name
    materials: the material list as loaded from the material file (not augmented)
@return: the material index
"""
def fallback_material_index(category: str, materials: list[dict]) -> int:
    name = category.lower()
    names = {m.get("name"): i for i, m in enumerate(materials)}
    default = rlr_material_index("default", materials)
    default = 0 if default is None else default

    def resolve(target):
        if target in names:
            return names[target]
        index = rlr_material_index(target, materials)
        return default if index is None else index

    for variant in (name.replace(" ", "_"), name.replace(" ", "-")):
        index = rlr_material_index(variant, materials)
        if index is not None:
            return index

    def stems(w):
        return (w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith("es") else w)

    def close_match(w, vocabulary, cutoff):
        #typos rarely change the first letter; requiring it keeps real words
        #from being "corrected" (heating -> seating)
        if w in _NOT_TYPOS:
            return None
        same_start = [v for v in vocabulary if v[0] == w[0]]
        return (difflib.get_close_matches(w, same_start, n=1, cutoff=cutoff) or [None])[0]

    #misspelt labels: treat "widnow frame" exactly like "window frame"
    labels = sorted({
        l.lower() for m in materials for l in m.get("labels", []) if l.isalpha() and len(l) >= 4
    })
    tokens = re.findall(r"[a-z]+", name)
    fixed = [
        (close_match(w, labels, 0.8) or w)
        if len(w) >= 4 and not any(f in _FALLBACK_KEYWORDS for f in stems(w)) else w
        for w in tokens
    ]
    if fixed != tokens:
        index = rlr_material_index(" ".join(fixed), materials)
        if index is not None:
            return index

    for keyword, target in _FALLBACK_KEYWORDS.items():
        if " " in keyword and keyword in name.replace("-", " "):
            return resolve(target)

    words = [
        w for part in _PHRASE_SPLIT.split(name) for w in reversed(re.findall(r"[a-z]+", part))
    ]
    for allowed in (_MATERIAL_WORDS, _FALLBACK_KEYWORDS):
        for w in words:
            for f in stems(w):
                if f in allowed and f in _FALLBACK_KEYWORDS:
                    return resolve(_FALLBACK_KEYWORDS[f])

    #misspelt keywords ("pilllow", "dishawasher"); stricter, words are real english
    keywords = sorted(k for k in _FALLBACK_KEYWORDS if " " not in k)
    for w in words:
        close = close_match(w, keywords, 0.85) if len(w) >= 5 else None
        if close:
            return resolve(_FALLBACK_KEYWORDS[close])

    return default


"""
@brief: give every category of a scene an explicit material. Categories no label
matches are appended as labels to their fallback material; those that end on the
default material are left to RLR's own default. Because RLR counts every label hit,
a short added label can tip a longer category ("tin" inside "painting"), so any
category whose intended material is no longer a strict winner gets another vote.
Categories the original file already matched keep their material.
@param:
    materials: the material list, modified in place
    categories: the category names present in the scene
@return: (categories without an original match, of those left on the default,
categories that still do not resolve to their intended material)
"""
def augment_material_labels(materials: list[dict], categories) -> tuple[int, int, int]:
    original = copy.deepcopy(materials)
    default = rlr_material_index("default", original)
    default = 0 if default is None else default

    enforced, num_fallback, num_default = {}, 0, 0
    for category in sorted({c.lower() for c in categories if c}):
        index = rlr_material_index(category, original)
        if index is None:
            num_fallback += 1
            index = fallback_material_index(category, original)
            if index == default:
                num_default += 1
                continue
            materials[index].setdefault("labels", []).append(category)
            enforced[category] = index
        elif _is_strict_winner(category, original, index):
            enforced[category] = index

    for _ in range(5):
        unstable = [c for c, i in enforced.items() if not _is_strict_winner(c, materials, i)]
        if not unstable:
            break
        for category in unstable:
            materials[enforced[category]]["labels"].append(category)
    unstable = [c for c, i in enforced.items() if not _is_strict_winner(c, materials, i)]

    return num_fallback, num_default, len(unstable)


"""
@brief: raised when habitat-sim does not simulate audio sources where they are put.
This stops the whole build, because every scene would be simulated wrongly.
"""
class AudioSourcePositionError(RuntimeError):
    pass

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
        self.audio, file_sample_rate = sf.read(raw_audio_path)
        self.sensor_name = sensor_name
        if sample_rate != 44100.0: 
            raise ValueError("Sample rate not 44100.0 in AudioCue instantiation")
        #sources are mono
        if self.audio.ndim > 1:
            self.audio = self.audio.mean(axis=1)
        #resample to the simulation rate (EARS is recorded at 48 kHz); without this
        #48 kHz speech plays back ~9% slow and pitched down
        if file_sample_rate != sample_rate:
            g = np.gcd(int(sample_rate), int(file_sample_rate))
            self.audio = ss.resample_poly(
                self.audio, int(sample_rate) // g, int(file_sample_rate) // g
            )
        self.num_samples = len(self.audio)
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
    #minimum distance between a speaker and the listener (m)
    MIN_SOURCE_DISTANCE = 0.5
    #attempts at a valid placement / audible configuration before a scene is skipped
    MAX_PLACEMENT_TRIES = 20
    #placements whose weakest RIR is further below the strongest (dB) are resampled:
    #those speakers are sealed off from the listener or placed outside the mesh
    MAX_RIR_RANGE_DB = 30.0
    #navmesh islands smaller than this (m^2) mark the scene as invalid
    MIN_ISLAND_AREA = 5.0

    """
    __init__

    @brief: load all mesh and audio source paths. 
    @param:
        max_level_diff_db: no speaker ends up more than this far below the loudest 
        speaker at the listener (see compute_audibility_gains)
        rlr_threads: CPU threads habitat-sim's ray tracer may use per simulation
        instance_timeout: seconds allowed per instance before a scene process is 
        killed and the scene skipped
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
        filter_by_semantic_scene: bool = True,
        max_level_diff_db: float = 6.0,
        rlr_threads: int = 1,
        instance_timeout: float = 1800.0
    ) -> None: 

        self.audio_absorption_file = audio_absorption_file
        self.dataset_config_file = dataset_config_file

        self.data_dirs = data_dirs
        self.data_paths = []
        for p_dir in self.data_dirs: 
            if not os.path.isdir(p_dir): continue
            self.data_paths.extend([
                os.path.join(p_dir, dir) for dir in os.listdir(p_dir)
                if dir.lower().endswith((".wav", ".flac"))
            ])
        #recordings grouped by speaker folder, so a mix never has one voice twice
        self.speaker_paths = {}
        for p in self.data_paths:
            self.speaker_paths.setdefault(os.path.dirname(p), []).append(p)

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
        self.max_level_diff_db = max_level_diff_db
        self.rlr_threads = rlr_threads
        self.instance_timeout = instance_timeout
        #set once a scene passed verify_audio_source_positions
        self.source_positions_verified = True

    def scene_contains_semantics(self, scene_path: str) -> bool:
        scene_files = os.listdir(scene_path)
        path_base = os.path.basename(scene_path).split("-")[-1]
        semantic_name = path_base + ".semantic.glb"
        for f in scene_files: 
            if f == semantic_name: return True
        return False

    """
    @brief: where this process keeps the scene's augmented material file
    """
    def scene_material_path(self) -> str:
        return os.path.join(
            tempfile.gettempdir(), f"spatialsoundmix_materials_{os.getpid()}.json"
        )

    """
    @brief: write this scene's copy of the material file, with a fallback material for
    every HM3D category the original labels miss (see augment_material_labels).
    @return: the path of the written file
    """
    def write_scene_audio_materials(
        self,
        sim: habitat_sim.simulator.Simulator
    ) -> str:
        with open(self.audio_absorption_file) as f:
            database = json.load(f)

        categories = {
            obj.category.name() for obj in sim.semantic_scene.objects
            if obj is not None and obj.category is not None
        }
        num_fallback, num_default, num_unstable = augment_material_labels(
            database["materials"], categories
        )

        path = self.scene_material_path()
        with open(path, "w") as f:
            json.dump(database, f)
        print(
            f"materials: {len(categories)} categories, {num_fallback} without a label match, "
            f"{num_fallback - num_default} given a fallback material, "
            f"{num_unstable} not resolving as intended"
        )
        return path


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
    @return:
        the sensor observations (one RIR per source) for the new placement, or
        None if the placement is invalid (see audio_rirs_valid)
    """
    def set_random_audio_source_positions(
        self,
        sim: habitat_sim.simulator.Simulator,
        num_sources: int,
        island_index: int, 
        max_distance: float = float("inf") 
    ) -> dict:
        #set position of agent to random point
        agent = sim.get_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_pos = sim.pathfinder.get_random_navigable_point(island_index=island_index)
        if not np.all(np.isfinite(agent_pos)):
            return None
        agent_state.position = agent_pos
        agent.set_state(agent_state)
        listener_pos = np.array(
            agent._sensors["audio_source_0"].node.absolute_translation, dtype=np.float32
        )

        #set position of sensors to random points
        for i in range(num_sources): 
            for _ in range(self.MAX_PLACEMENT_TRIES):
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
                if not np.all(np.isfinite(valid_floor_pos)):
                    continue

                initial_pos = self.set_random_elevation(sim, valid_floor_pos)
                #keep speakers out of the listener's head
                if np.linalg.norm(initial_pos - listener_pos) >= self.MIN_SOURCE_DISTANCE:
                    break
            else:
                return None

            sensor = agent._sensors[f"audio_source_{i}"]
            sensor.setAudioSourceTransform(initial_pos)

        #a single ray tracing pass; its RIRs are reused for generation,
        #so no separate warmup step is needed
        obs = sim.get_sensor_observations()
        return obs if self.audio_rirs_valid(obs, num_sources) else None

    """
    @brief: check that every source produced a usable RIR: present, finite, non-silent,
    and no more than MAX_RIR_RANGE_DB below the strongest one. Quieter RIRs come from
    speakers sealed off from the listener or placed outside the mesh.
    """
    def audio_rirs_valid(
        self,
        obs: dict,
        num_sources: int,
        num_channels: int = 2
    ) -> bool:
        energies = []
        for i in range(num_sources):
            rir = np.asarray(obs.get(f"audio_source_{i}", []), dtype=np.float64)
            if (
                rir.ndim != 2 or rir.shape[0] < num_channels or rir.shape[1] == 0
                or not np.all(np.isfinite(rir))
            ):
                return False
            energies.append(np.sum(rir ** 2))

        if min(energies) <= 0.0:
            return False
        return 10.0 * np.log10(max(energies) / min(energies)) <= self.MAX_RIR_RANGE_DB

    """
    @brief: make sure habitat-sim simulates sources where they are placed.
    habitat-sim >= 0.3.1 hands the ray tracer lastSourcePos_[1], [2], [3] in
    AudioSensor::runSimulation (src/esp/sensor/AudioSensor.cpp), i.e. (y, z, <garbage>)
    instead of (x, y, z), which puts most speakers outside the house where they are
    barely heard. A source 0.3 m above the listener is compared with the same source
    moved 200 m along x: a correct build hears the far one tens of dB quieter, the
    broken build ignores x and returns the same RIR.
    @note: only sensor 0 is run, through the same calls habitat-sim's own sensor
    wrapper makes.
    """
    def verify_audio_source_positions(
        self,
        sim: habitat_sim.simulator.Simulator,
        island_index: int,
        far_offset: float = 200.0
    ) -> None:
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
            agent_state.position = sim.pathfinder.get_random_navigable_point(
                island_index=island_index
            )
            if not np.all(np.isfinite(agent_state.position)):
                continue
            agent.set_state(agent_state)

            #the probe source needs head room above the listener
            listener_pos = np.array(sensor.node.absolute_translation, dtype=np.float32)
            ray = habitat_sim.geo.Ray(listener_pos, np.array([0.0, 1.0, 0.0]))
            hits = sim.cast_ray(ray).hits
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
            "again. To build anyway, set builder.source_positions_verified = True."
        )


    def get_largest_island_for_scene(
        self, 
        sim: habitat_sim.simulator.Simulator
    )-> int: 
        pf = sim.pathfinder 
        if pf.num_islands == 0:
            raise ValueError("navmesh has no navigable islands")
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
        seed: seed for habitat-sim (defaults to self.seed)
    @return: 
        the new simulator
    """
    def initialize_sim_scene(
        self,
        num_sources: int, 
        mesh_path: str,
        sim: habitat_sim.Simulator = None,
        max_distance: float = 10.0,
        seed: int = None
    ) -> habitat_sim.simulator.Simulator: 
        seed = self.seed if seed is None else seed
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
        backend_cfg.random_seed = seed

        #setup acoustics
        acoustics_cfg = habitat_sim.RLRAudioPropagationConfiguration()
        acoustics_cfg.enableMaterials = True 
        acoustics_cfg.temporalCoherence = False
        acoustics_cfg.threadCount = self.rlr_threads

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
        sim.seed(seed)

        if len(sim.semantic_scene.objects) == 0: 
            print("semantics not loaded properly in scene initialization")
            raise ValueError("Semantics not loaded")

        #initialize navmesh
        navmesh_path = mesh_path[:-4] + ".navmesh"
        nav_mesh_exit = sim.pathfinder.load_nav_mesh(navmesh_path)

        if not nav_mesh_exit: 
            raise ValueError("attempted loading an invalid navmesh")

        island_idx = self.get_largest_island_for_scene(sim)
        island_area = sim.pathfinder.island_area(island_idx)
        if island_area < self.MIN_ISLAND_AREA: 
            raise ValueError(f"largest navmesh island is only {island_area:.1f} m^2")

        #load acoustic file for each sensor: a per-scene copy of the material file
        #with fallback materials for the HM3D categories it does not label
        material_file = self.write_scene_audio_materials(sim)
        agent = sim.get_agent(0)
        for sensor_name, sensor in agent._sensors.items(): 
            #only audio sensors are made so only audio sensors are accessed
            sensor.setAudioMaterialsJSON(material_file)

        if not self.source_positions_verified:
            self.verify_audio_source_positions(sim, island_idx)
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
    @brief: pick one recording per source, each from a different speaker folder,
    so no voice talks over itself in a mix.
    """
    def sample_source_paths(self, num_sources: int) -> list[str]:
        if num_sources > len(self.speaker_paths):
            raise ValueError(
                f"{num_sources} sources requested but only {len(self.speaker_paths)} speakers found"
            )
        speakers = random.sample(sorted(self.speaker_paths), num_sources)
        return [random.choice(self.speaker_paths[s]) for s in speakers]

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
    @brief: guarantee that every speaker is audible in the mix. Each source's level at
    the listener is measured as the RMS of its dry audio convolved with its RIR, over
    the samples where the source is active, summed over both ears. Any source more
    than max_level_diff_db below the loudest one is raised to exactly that bound;
    sources already inside the window keep their natural distance / occlusion level.
    The gain is the same for both ears, so interaural and direct-to-reverberant cues
    are untouched.
    @param:
        audio_sequence: the sequenced cues
        rirs: {sensor name: (channels, samples) RIR}
    @return:
        {sensor name: linear gain}, or None if a source never reaches the listener
    """
    def compute_audibility_gains(
        self,
        audio_sequence: AudioSequence,
        rirs: dict
    ) -> dict:
        levels = []
        for i, q in enumerate(audio_sequence.cues):
            rir = rirs[f"audio_source_{i}"]
            active = q.activity_mask.astype(bool)
            if not active.any():
                active = np.ones_like(active)
            energy = sum(
                np.mean(ss.fftconvolve(q.audio, rir[c])[:q.num_samples][active] ** 2)
                for c in range(rir.shape[0])
            )
            levels.append(10.0 * np.log10(energy) if energy > 0 else -np.inf)

        levels = np.asarray(levels)
        if not np.all(np.isfinite(levels)):
            return None
        boost_db = np.maximum(0.0, levels.max() - self.max_level_diff_db - levels)

        return {f"audio_source_{i}": 10.0 ** (b / 20.0) for i, b in enumerate(boost_db)}

    """
    @breif: Generates one instance of simulated audio with static positioning. 
    The current setup is for generating BSC data but can be easily altered
    by simply returning the target audio separation as well. 

    @param: 
        sim: the habit_sim simulator for ray tracing
        audio_sequence: the audio sequence for enumerating time slices
        num_channels: the number of channels of the sim
        obs: sensor observations for the current placement; computed if None
    @return: 
        the instance dict, or None if some speaker cannot be made audible
    """
    def generate_stationary_data_instance(
        self, 
        sim: habitat_sim.simulator.Simulator,
        audio_sequence: AudioSequence, 
        num_channels: int = 2,
        obs: dict = None
    ) -> dict: 
        #only get observations once for stationary audio
        if obs is None:
            obs = sim.get_sensor_observations()
        #convert each RIR once rather than at every step
        rirs = {
            f"audio_source_{i}": np.asarray(obs[f"audio_source_{i}"], dtype=np.float64)
            for i in range(len(audio_sequence.cues))
        }

        #per-source gains that keep every speaker audible
        gains = self.compute_audibility_gains(audio_sequence, rirs)
        if gains is None:
            return None

        mix_audio = np.zeros((num_channels, audio_sequence.max_samples))
        source_counts = np.zeros((num_channels, audio_sequence.max_samples))

        prog_bar = tqdm(
            range(len(audio_sequence)), 
            desc="Generating Sequence", 
            leave=False
        )

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

                rir = rirs[s_name]
                s_audio = s_audio * gains[s_name]

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
        seed: seed for habitat-sim (defaults to self.seed)
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
        max_distance: float = 10.0,
        seed: int = None
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

        if not mesh_path:
            raise ValueError(f"no .basis.glb mesh in {mesh_dir}")
        if not os.path.isfile(mesh_path[:-4] + ".navmesh"):
            raise ValueError(f"no navmesh next to {mesh_path}")

        sim  = self.initialize_sim_scene(
            num_sensors, mesh_path, sim=sim, max_distance=max_distance, seed=seed
        )

        #create folder for examples
        write_dir = os.path.join(data_write_path, instance_name)
        os.makedirs(write_dir, exist_ok=True)

        """
        generate num_states unique configurations in this room
        with this AudioSensor count.
        """
        for i in tqdm(range(num_states)): 
            #retry placements / recordings until every speaker is audible
            data_instance = None
            for _ in range(self.MAX_PLACEMENT_TRIES):
                island_idx = self.get_largest_island_for_scene(sim)
                obs = self.set_random_audio_source_positions(
                    sim, num_sensors, island_idx, max_distance=max_distance
                )
                if obs is None:
                    continue

                source_paths = self.sample_source_paths(num_sensors)
                source_sequence, sr = self.create_audio_cue_sequence(
                    source_paths, step_duration=step_duration,
                )

                print(f"   -> Generating instnance {i} for scene at : {mesh_path}")
                data_instance = self.generate_stationary_data_instance(
                    sim, source_sequence, obs=obs
                )
                if data_instance is not None:
                    break
            if data_instance is None:
                raise RuntimeError(
                    f"no configuration with every speaker audible after "
                    f"{self.MAX_PLACEMENT_TRIES} tries"
                )
            print("    -> Generation done")


            #make instance write dir: written to a hidden temporary folder that is
            #renamed once complete, so an interrupted run never leaves half an instance
            instance_write_dir = os.path.join(write_dir, f"instance_{i}")
            tmp_write_dir = os.path.join(write_dir, f".instance_{i}.tmp")
            shutil.rmtree(tmp_write_dir, ignore_errors=True)
            os.makedirs(tmp_write_dir)

            write_path_audio = os.path.join(tmp_write_dir, f"mix_audio.wav")
            write_path_counts = os.path.join(tmp_write_dir, f"counts.npy")
            write_path_source = os.path.join(tmp_write_dir, f"source")

            #write audio and source counts
            sf.write(write_path_audio, data_instance['mix_audio'].T, int(sr))
            np.save(write_path_counts, data_instance['source_counts'])

            if return_sources: 
                for k, src in enumerate(data_instance["sources"]):
                    path = write_path_source + f"_{k}.wav"
                    sf.write(path, src.T, int(sr))

            shutil.rmtree(instance_write_dir, ignore_errors=True)
            os.rename(tmp_write_dir, instance_write_dir)

        return sim

    """
    @brief: generates all dynamic instance of audio for a scene
    """
    def generate_dynamic_instances_for_scene(self): 
        pass

    """
    @brief: True when every instance of a scene has been written
    """
    def scene_is_complete(self, write_dir: str, num_states: int) -> bool:
        return all(
            os.path.isfile(os.path.join(write_dir, f"instance_{k}", name))
            for k in range(num_states) for name in ("mix_audio.wav", "counts.npy")
        )

    """
    @brief: body of the process build_dataset starts for one scene. The process owns
    the simulator for its scene only, so memory and threads cannot pile up across
    scenes and a crash inside habitat-sim ends only this scene.
    """
    def _scene_process(
        self,
        scene_idx: int,
        mesh_dir: str,
        num_sensors: int,
        num_states: int,
        instance_name: str,
        data_write_path: str,
        return_sources: bool,
        step_duration: float,
        mesh_type: str,
        max_distance: float
    ) -> None:
        #seeded per scene, so a scene comes out the same whatever ran before it
        random.seed(self.seed + scene_idx)
        #start clean: drops partial output of an interrupted earlier run
        shutil.rmtree(os.path.join(data_write_path, instance_name), ignore_errors=True)
        try:
            sim = self.generate_stationary_instances_for_scene(
                mesh_dir, num_sensors, num_states, instance_name,
                data_write_path, return_sources, step_duration, mesh_type,
                max_distance=max_distance, seed=self.seed + scene_idx
            )
            sim.close()
        except AudioSourcePositionError as e:
            print(f"\n{e}", file=sys.stderr, flush=True)
            sys.exit(_FATAL_EXIT_CODE)
        finally:
            if os.path.exists(self.scene_material_path()):
                os.remove(self.scene_material_path())

    """
    build_dataset

    @breif: builds the whole train dataset
    @Note: Need to edit audio source writing so that audio 
    sources are same length as audio mixes for slicing. In essence
    audio sources need to be the target split of audio mixes

    Each scene runs in its own process, one at a time. A scene that raises, segfaults,
    is killed for memory or runs past its timeout is skipped and its partial output 
    removed, so bad scenes no longer need to be listed by hand. Scenes already written
    are skipped, so an interrupted build resumes where it stopped.
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
        #spawn, not fork: a fresh interpreter per scene, no inherited GL / thread state
        ctx = mp.get_context("spawn")
        skipped = {}
        prog_bar = tqdm(
            enumerate(self.mesh_paths), 
            total=len(self.mesh_paths),
            desc="Building Scene Data"
        )
        for i, path in prog_bar:
            instance_name = f"scene_{i}"
            write_dir = os.path.join(data_write_path, instance_name)
            if self.scene_is_complete(write_dir, num_states_per_scene):
                continue
            num_sensors = random.Random(self.seed + i).randint(lo, hi)

            proc = ctx.Process(
                target=self._scene_process,
                args=(
                    i, path, num_sensors, num_states_per_scene, instance_name,
                    data_write_path, return_sources, step_duration, mesh_type, max_distance
                )
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
