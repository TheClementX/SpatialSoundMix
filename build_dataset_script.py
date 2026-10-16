"""
build_dataset_script.py

"""

from static_dataset_builder import *
import os 

#testing paths and configuration
data_base_dir = "/home/clem3nti/projects/SpatialSoundMix/data/ears_dataset"
DATA_DIRS = [os.path.join(data_base_dir, dir) for dir in os.listdir(data_base_dir)]
MESH_DIRS = ["/home/clem3nti/projects/SpatialSoundMix/data/versioned_data/hm3d-0.2/hm3d/train"]
DATA_WRITE_PATH = "/home/clem3nti/projects/SpatialSoundMix/data_test"
STEP_DURATION = 0.2
SAMPLE_RATE = 44100.0
SEED = 42

#dataset_builder parameters
NUM_STATES_PER_SCENE = 5
RETURN_SOURCES = True
MAX_MIN_SENSORS = (4, 8)

def main(): 
    dataset_builder = SpatialSoundMixDatasetBuilder(
        DATA_DIRS, MESH_DIRS, STEP_DURATION, SAMPLE_RATE, SEED
    )
    dataset_builder.build_dataset(
        NUM_STATES_PER_SCENE, MAX_MIN_SENSORS, DATA_WRITE_PATH, 
        RETURN_SOURCES, STEP_DURATION, "glb"
    )

if __name__ == "__main__": 
    main()