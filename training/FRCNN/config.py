import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "kitti_object")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

BATCH_SIZE = 4
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
EPOCHS = 25
NUM_CLASSES = 6 
WARMUP_EPOCHS = 1
LR_STEP_SIZE = 7
LR_GAMMA = 0.5
