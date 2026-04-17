import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_PATH, "data", "kitti_raw")
SEQ_PATH = os.path.join(
    RAW_DATA_PATH,
    "2011_09_26_drive_0059_sync",
    "2011_09_26",
    "2011_09_26_drive_0059_sync",
    "image_02",
    "data",
)
CALIB_FILE = os.path.join(
    RAW_DATA_PATH,
    "2011_09_26_calib",
    "2011_09_26",
    "calib_cam_to_cam.txt",
)

OUTPUT_PATH = os.path.join(BASE_PATH, "output_frames")
VIDEO_OUTPUT = os.path.join(BASE_PATH, "output_video.mp4")

OUTPUT_FPS = 10  
NUM_CLASSES = 6   # Background, Car, Van, Truck, Pedestrian, Cyclist
SCORE_THRESH = 0.65
NMS_IOU_THRESH = 0.35
