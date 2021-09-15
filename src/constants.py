import os

# Data Storage and Transform
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # HTR
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed")
TARGET_FOLDER_PATH = os.path.join(ROOT_DIR, "target")
RESULTS_FOLDER_PATH = os.path.join(TARGET_FOLDER_PATH, "results")
DEFAULT_SEGMENTATION_LEVEL = "word"
DEFAULT_PARTITIONING = {'train': "train.txt", 'validation': "val.txt", 'test': "test.txt"}

SAMPLES_PER_TFRECORD = 1024
INPUT_SIZE = (1024, 128, 1)
VOCABULARY_SIZE = 128
STORAGE_FORMAT = "tfrecord"
IMAGE_FORMAT = "png"

# Preprocessing
ROTATION_RANGE = 0
SCALE_RANGE = 0
HEIGHT_SHIFT_RANGE = 0
WIDTH_SHIFT_RANGE = 0
DILATION_RANGE = 1
ERODE_RANGE = 1
MAX_SEQUENCE_LENGTH = 128

# Machine Learning
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MONITOR_PARAM = "val_loss"
STOP_TOLERANCE = 20
REDUCE_TOLERANCE = 15
COOLDOWN = 0
EPOCHS = 1000
