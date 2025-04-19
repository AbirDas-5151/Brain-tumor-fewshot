# Base paths (update as needed)
brats_path = '/path/to/BraTS 2020'
crystal_path = '/path/to/Crystal Clean'

# Config: sample limits for debugging
NUM_SAMPLES_BRA_TS_TRAIN = 30
NUM_SAMPLES_BRA_TS_VAL = 20
NUM_SAMPLES_CRYSTAL_NORMAL = 30
NUM_SAMPLES_PER_TUMOR_CLASS = 30

# Image size for resizing
IMAGE_SIZE = (224, 224)

# Split ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
