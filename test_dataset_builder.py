"""
test_dataset_builder.py

@brief: scripting to test the various functionalities of the static dataset 
builder functionality. 
"""

from static_dataset_builder import *
import os 

#testing paths and configuration
base_dir = "/home/clem3nti/projects/SpatialSoundMix/data/ears_dataset"
DATA_DIRS = [os.path.join(base_dir, dir) for dir in os.listdir(base_dir)]
MESH_DIRS = ["/home/clem3nti/projects/SpatialSoundMix/data/versioned_data/hm3d-0.2/hm3d/minival"]
STEP_DURATION = 0.2
SAMPLE_RATE = 44100.0
SEED = 42

DATA_WRITE_PATH = "/home/clem3nti/projects/SpatialSoundMix/data_test"
INSTANCE_NAME = "test_instance"

def test_static_instance_generation():
    dataset_builder = SpatialSoundMixDatasetBuilder(
        DATA_DIRS, MESH_DIRS, STEP_DURATION, SAMPLE_RATE, SEED
    )

    dataset_builder.generate_stationary_instances_for_scene(
        mesh_dir = dataset_builder.mesh_paths[0], 
        num_sensors = 8, 
        num_states = 1, 
        instance_name = INSTANCE_NAME, 
        data_write_path = DATA_WRITE_PATH, 
        step_duration = STEP_DURATION, 
        mesh_type = "glb"
    )

def main(): 
    test_static_instance_generation()

if __name__ == "__main__": 
    main()