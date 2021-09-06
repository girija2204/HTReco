import os

# Data Storage and Transform
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_PER_TFRECORD = 128
INPUT_SIZE = (1024, 128, 3)
STORAGE_FORMAT = "tfrecord"

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
