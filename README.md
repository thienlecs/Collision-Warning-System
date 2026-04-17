# Collision Warning System - KITTI Dataset

This project implements a comprehensive collision warning system (CWS) using Object Detection (Faster R-CNN), Multi-object Tracking (SORT), Depth Estimation, and Time-to-Collision (TTC) calculations.

## Dataset Information

All datasets are sourced from the [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/index.php).

- **Training Dataset**: Download from the **"2D Object"** section for model fine-tuning, then extract it to data/kitti_object.
- **Inference Dataset**: Download from the **"Raw Data"** section for real-world simulation (continuous video sequences), then extract it to data/kitti_raw.
## Getting Started

### 1. Environment Setup
Install the necessary dependencies:
```bash
pip install torch torchvision numpy opencv-python tqdm filterpy Pillow
```

### 2. Model Training
To train the model on the KITTI 2D Object dataset:
1. Update paths and hyperparameters in `training/config.py`.
2. Execute the training script:
```bash
python -m training.FRCNN.train
```
Skip training by downloading a sample model [Here](https://drive.google.com/drive/folders/1-XDiAl4l6JwK2WlM63dkqhSTyn5znF1U?usp=sharing) and placing it in the checkpoints/ directory.

### 3. Running Inference
To run the system on an image sequence (dataset required):
```bash
python cws_main.py
```
Processed outputs, including bounding boxes, distance estimates, class labels, tracking IDs and collision alerts (Safe/Warning/Danger), will be visualized and saved.
