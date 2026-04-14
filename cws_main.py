import os
import glob
import shutil
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image, ImageDraw
import numpy as np
import torchvision.ops as ops

from utils.calib import Calibration
from utils.sort import Sort
from utils.depth_estimation import DepthEstimator
from utils.ttc import TTCCalculator
from utils.roi_filter import ROIFilter

BASE_PATH = "d:/CWS"
RAW_DATA_PATH = os.path.join(BASE_PATH, "data/kitti_raw")
SEQ_PATH = os.path.join(RAW_DATA_PATH, "2011_09_26_drive_0057_sync/2011_09_26/2011_09_26_drive_0057_sync/image_02/data")
CALIB_FILE = os.path.join(RAW_DATA_PATH, "2011_09_26_calib/2011_09_26/calib_cam_to_cam.txt")
OUTPUT_PATH = os.path.join(BASE_PATH, "output_frames")

def cleanup_output_dir():
    if os.path.exists(OUTPUT_PATH):
        for f in os.listdir(OUTPUT_PATH):
            os.remove(os.path.join(OUTPUT_PATH, f))
    else:
        os.makedirs(OUTPUT_PATH)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_latest_checkpoint():
    d = os.path.join(BASE_PATH, "checkpoints")
    if not os.path.exists(d): return None
    files = glob.glob(os.path.join(d, "*.pth"))
    if not files: return None
    return max(files, key=os.path.getmtime)

def main():
    cleanup_output_dir()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    num_classes = 4 # Background, Vehicle, Pedestrian, Cyclist
    weights_path = get_latest_checkpoint()

    model = get_model(num_classes)
    if os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print("Error: Weights file not found. Please train the model first.")
        return
        
    model.to(device)
    model.eval()

    # Init Modules
    calib = Calibration(CALIB_FILE)
    focal_length = calib.get_focal_length_x(camera_id=2)
    print(f"Camera 02 Focal length (f_x): {focal_length}")

    tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.3)
    depth_estimator = DepthEstimator(focal_length_x=focal_length)
    ttc_calculator = TTCCalculator(fps=10.0) 
    
    roi_filter = ROIFilter(image_width=1242, image_height=375)

    transform = T.Compose([T.ToTensor()])
    
    image_files = sorted(glob.glob(os.path.join(SEQ_PATH, "*.png")))
    if not image_files:
        print(f"Image sequence not found at {SEQ_PATH}")
        return

    print("Starting CWS processing pipeline...")
    for idx, img_path in enumerate(image_files):
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).to(device)
        
        with torch.no_grad():
            prediction = model([img_tensor])[0]
            
        # Filter low confidence scores
        keep_score_idx = prediction['scores'] > 0.5
        b_boxes = prediction['boxes'][keep_score_idx]
        b_scores = prediction['scores'][keep_score_idx]
        b_labels = prediction['labels'][keep_score_idx]

        # Apply Class-Agnostic NMS
        # This removes overlapping boxes regardless of their predicted class
        keep_nms_idx = ops.nms(b_boxes, b_scores, iou_threshold=0.45)
        
        b_boxes = b_boxes[keep_nms_idx].cpu().numpy()
        b_scores = b_scores[keep_nms_idx].cpu().numpy()
        b_labels = b_labels[keep_nms_idx].cpu().numpy()

        detections = []
        for i in range(len(b_boxes)):
            x1, y1, x2, y2 = b_boxes[i]
            detections.append([x1, y1, x2, y2, b_scores[i], b_labels[i]])

        # Handle no detections
        detections = np.array(detections) if len(detections) > 0 else np.empty((0,6))
        
        tracked_objects = tracker.update(detections)
        
        draw = ImageDraw.Draw(img_pil)
        
        draw.line(roi_filter.get_polygon_for_drawing(), fill="yellow", width=2)
        
        color_map = {"Danger": "red", "Warning": "orange", "Safe": "lime", "Unknown": "white", "Cross-Traffic": "cyan"}

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id, obj_class = obj
            
            in_path = roi_filter.is_in_path([x1, y1, x2, y2])
            depth = depth_estimator.estimate_depth([x1, y1, x2, y2], obj_class)
            
            if in_path:
                ttc, alert_level = ttc_calculator.update_and_calculate(obj_id, depth, [x1, y1, x2, y2])
                color = color_map.get(alert_level, "white")
                
                if alert_level == "Cross-Traffic":
                    ttc_str = "Inf"
                    text = f"ID:{int(obj_id)} | Cross-Traffic"
                else:
                    ttc_str = f"{ttc:.1f}s" if ttc != float('inf') else "Inf"
                    text = f"ID:{int(obj_id)} | Z:{depth:.1f}m | TTC:{ttc_str}"
            else:
                color = "gray"
                text = f"ID:{int(obj_id)} | Out of Path"
                ttc_calculator.update_and_calculate(obj_id, depth, [x1, y1, x2, y2])
            
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
            
            bg_width = max(len(text) * 6, 40)
            draw.rectangle(((x1, max(0, y1-15)), (x1 + bg_width, max(0, y1))), fill=color)
            draw.text((x1 + 2, max(0, y1-14)), text, fill="black" if color != "gray" else "white")

        output_file = os.path.join(OUTPUT_PATH, os.path.basename(img_path))
        img_pil.save(output_file)
        
        if (idx + 1) % 20 == 0:
            print(f"Processed frame {idx+1}/{len(image_files)}")

    print(f"DONE! {len(image_files)} frames have been saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
