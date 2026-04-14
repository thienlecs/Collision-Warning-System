import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "kitti_object")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

BATCH_SIZE = 4
LR = 0.0005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
EPOCHS = 5 # Obsevered that loss is not decreasing after 5 epochs
NUM_CLASSES = 4 # Background, Vehicle, Pedestrian, Cyclist
