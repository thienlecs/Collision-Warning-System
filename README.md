# Collision Warning System - KITTI Dataset

This project implements a collision warning system using Object Detection (Faster R-CNN), Multi-object Tracking (SORT), Depth Estimation, and Time-to-Collision (TTC) calculations.

## Project Structure
- training/: Contains the training pipeline logic.
  - dataset.py: KITTI dataset loader.
  - model.py: Model architecture definition.
  - config.py: Hyperparameter management.
  - train.py: Training script.
- utils/: Supporting modules for the inference pipeline.
  - depth_estimation.py: Distance estimation based on bounding box geometry.
  - ttc.py: Time-to-collision logic and alert leveling.
  - roi_filter.py: Trajectory-based filtering to ignore off-path objects.
  - sort.py: Multi-object tracking implementation.
- cws_main.py: Main entry point for inference and visualization.

## Quick Start Walkthrough

### 1. Environment Setup
Install the necessary dependencies.

### 2. Model Training
To train or fine-tune the model on the KITTI dataset:
1. Configure settings in training/config.py.
2. Run the training command from the root directory:
```bash
python -m training.train
```
### 3. Running the System (Inference)
Execute the main pipeline:
```bash
python cws_main.py
```
Processed frames will be saved in the output_frames/ directory.