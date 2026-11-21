"""
build_dataset_script_cl.py

Builds the static SpatialSoundMix dataset. Edit the settings below and run:
    python build_dataset_script_cl.py
"""

import os
from static_dataset_builder_cl import SpatialSoundMixDatasetBuilder

"""
DATA PATH VARIABLES ============================================= 
"""
DATA_BASE_DIR = "/home/clem3nti/projects/SpatialSoundMix/data/ears_dataset"
DATA_DIRS = [os.path.join(DATA_BASE_DIR, d) for d in sorted(os.listdir(DATA_BASE_DIR))]
MESH_DIRS = ["/home/clem3nti/projects/SpatialSoundMix/data/hm3d_default/versioned_data/hm3d-0.2/hm3d/train"]
DATA_WRITE_PATH = "/home/clem3nti/projects/SpatialSoundMix/r_data"
AUDIO_ABSORPTION_FILE = "/home/clem3nti/projects/SpatialSoundMix/data/mp3d_material_config.json"
DATASET_CONFIG_FILE = "/home/clem3nti/projects/SpatialSoundMix/data/hm3d_default/versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
"""
DATA PATH VARIABLES ============================================= 
"""

"""
SIMULATION VARIABLES ============================================= 
"""
SAMPLE_RATE = 44100.0
SEED = 42
RLR_THREADS = 4 #numper of cpu threads
"""
SIMULATION VARIABLES ============================================= 
"""

"""
DATA ATTRIBUTE VARIABLES ============================================= 
"""
NUM_STATES_PER_SCENE = 30 #number of instances generated
NUM_SOURCES_RANGE = (2, 2) #number of sources in mix
MAX_DISTANCE = 5.0 #max speaker distance
MAX_LEVEL_DIFF_DB = 4.0 # db difference between loudest speaker and everyother speaker

RETURN_SOURCES = True
#how the source is generated
#   "dry"    the clean recording, at the level it was recorded at
#   "direct" the recording convolved with the direct path of its RIR (no reverb), at
#            the mix's scale
#   "full"   the recording convolved with its full RIR (reverberant), at the mix's
#            scale; the full targets of a mix sum to the mix
SOURCE_TYPE = "full"
#True: targets are binaural, 2 channels (left / right ear) like the mix; a dry
#target is the same signal in both channels.
#False: targets are mono, 1 channel, taken as MONO_CHANNEL says.
BINAURAL_SOURCES = False
#with BINAURAL_SOURCES False, what a mono direct / full target is:
#   "left"    the left ear
#   "right"   the right ear
#   "average" the average of both ears
#a dry target is the mono recording whatever this is set to
MONO_CHANNEL = "left"

"""
DATA ATTRIBUTE VARIABLES ============================================= 
"""

def main():
    builder = SpatialSoundMixDatasetBuilder(
        DATA_DIRS, MESH_DIRS, AUDIO_ABSORPTION_FILE, DATASET_CONFIG_FILE,
        sample_rate=SAMPLE_RATE,
        seed=SEED,
        source_type=SOURCE_TYPE,
        binaural_sources=BINAURAL_SOURCES,
        mono_channel=MONO_CHANNEL,
        max_level_diff_db=MAX_LEVEL_DIFF_DB,
        rlr_threads=RLR_THREADS,
    )
    builder.build_dataset(
        NUM_STATES_PER_SCENE, NUM_SOURCES_RANGE, DATA_WRITE_PATH,
        return_sources=RETURN_SOURCES,
        max_distance=MAX_DISTANCE,
    )

if __name__ == "__main__":
    main()
