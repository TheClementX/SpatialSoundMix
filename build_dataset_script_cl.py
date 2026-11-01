"""
build_dataset_script.py

"""

from static_dataset_builder_cl import *
import os 

#testing paths and configuration
data_base_dir = "/home/clem3nti/projects/SpatialSoundMix/data/ears_dataset"
DATA_DIRS = [os.path.join(data_base_dir, dir) for dir in os.listdir(data_base_dir)]
MESH_DIRS = ["/home/clem3nti/projects/SpatialSoundMix/data/versioned_data/hm3d-0.2/hm3d/train"]
MESH_DIRS_OG = ["/home/clem3nti/projects/SpatialSoundMix/data/hm3d_default/versioned_data/hm3d-0.2/hm3d/train"]
DATA_WRITE_PATH = "/home/clem3nti/projects/SpatialSoundMix/data_test"
DATA_WRITE_PATH_OG = "/home/clem3nti/projects/SpatialSoundMix/data_test_og"
STEP_DURATION = 0.2
SAMPLE_RATE = 44100.0
SEED = 42

#TODO set these file paths
AUDIO_ABSORPTION_FILE = "/home/clem3nti/projects/SpatialSoundMix/data/mp3d_material_config.json"
DATASET_CONFIG_FILE = "/home/clem3nti/projects/SpatialSoundMix/data/versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
DATASET_CONFIG_FILE_OG = "/home/clem3nti/projects/SpatialSoundMix/data/hm3d_default/versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

#dataset_builder parameters
NUM_STATES_PER_SCENE = 1
RETURN_SOURCES = True
MAX_MIN_SENSORS = (2, 4)
MAX_DISTANCE = 10.0
#no speaker is left more than this many dB below the loudest one in a mix
MAX_LEVEL_DIFF_DB = 6.0
#CPU threads for habitat-sim's ray tracer; scenes run one at a time, so this is the
#build's main CPU budget (raise it to go faster, lower it to leave cores free)
RLR_THREADS = 1

def main(): 
    dataset_builder = SpatialSoundMixDatasetBuilder(
        DATA_DIRS, MESH_DIRS_OG, 
        AUDIO_ABSORPTION_FILE, 
        DATASET_CONFIG_FILE_OG, 
        STEP_DURATION, SAMPLE_RATE, SEED,
        max_level_diff_db=MAX_LEVEL_DIFF_DB,
        rlr_threads=RLR_THREADS
    )
    dataset_builder.build_dataset(
        NUM_STATES_PER_SCENE, MAX_MIN_SENSORS, DATA_WRITE_PATH_OG, 
        RETURN_SOURCES, STEP_DURATION, "glb", MAX_DISTANCE
    )

#the guard is required: each scene runs in a spawned process that re-imports this file
if __name__ == "__main__": 
    main()
