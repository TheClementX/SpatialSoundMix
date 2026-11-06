"""
build_dataset_script.py

"""

from static_dataset_builder import *
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

def main(): 
    dataset_builder = SpatialSoundMixDatasetBuilder(
        DATA_DIRS, MESH_DIRS_OG, 
        AUDIO_ABSORPTION_FILE, 
        DATASET_CONFIG_FILE_OG, 
        STEP_DURATION, SAMPLE_RATE, SEED
    )
    dataset_builder.build_dataset(
        NUM_STATES_PER_SCENE, MAX_MIN_SENSORS, DATA_WRITE_PATH_OG, 
        RETURN_SOURCES, STEP_DURATION, "glb", MAX_DISTANCE
    )

if __name__ == "__main__": 
    main()