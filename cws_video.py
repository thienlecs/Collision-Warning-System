import os
import glob
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torchvision.ops as ops
import time

from utils.calib import Calibration
from utils.sort import Sort
from utils.depth_estimation import DepthEstimator
from utils.ttc import TTCCalculator
from utils.roi_filter import ROIFilter
from config import (
    BASE_PATH, CALIB_FILE,
    NUM_CLASSES, SCORE_THRESH, NMS_IOU_THRESH,
)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_best_checkpoint():
    d = os.path.join(BASE_PATH, "checkpoints")
    if not os.path.exists(d): return None
    best_path = os.path.join(d, "best_model.pth")
    if os.path.exists(best_path):
        return best_path
    files = glob.glob(os.path.join(d, "*.pth"))
    if not files: return None
    return max(files, key=os.path.getmtime)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    num_classes = NUM_CLASSES  # Defined in config.py
    weights_path = get_best_checkpoint()

    model = get_model(num_classes)
    if weights_path and os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print("Error: Weights file not found. Please train the model or provide best_model.pth first.")
        return
        
    model.to(device)
    model.eval()

    # Init Modules
    calib = Calibration(CALIB_FILE)
    focal_length = calib.get_focal_length_x(camera_id=2)
    print(f"Camera 02 Focal length (f_x): {focal_length}")

    # Synchronized parameters with cws_main.py
    tracker = Sort(max_age=1, min_hits=1, iou_threshold=0.35)
    depth_estimator = DepthEstimator(focal_length_x=focal_length)
    
    VIDEO_INPUT = "D:\CWS\demo_vid.mp4"
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_INPUT}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0 or np.isnan(video_fps):
        video_fps = 24.0 # Standard KITTI sequence rate
        
    ttc_calculator = TTCCalculator(fps=video_fps) 
    
    # KITTI default size, adjust if video has different aspect ratio
    roi_filter = ROIFilter(image_width=1242, image_height=375)
    transform = T.Compose([T.ToTensor()])

    print(f"Starting CWS processing pipeline on Video (Video Meta FPS: {video_fps})...")
    print("Press 'q' on the video window to stop.")

    frame_count = 0
    total_processing_time = 0

    while True:
        # Start timer BEFORE frame read for true system FPS
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        img_cv = frame.copy()
        
        # update ROI
        roi_filter.update(img_cv)

        # Process frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transform(img_pil).to(device)
        
        with torch.no_grad():
            prediction = model([img_tensor])[0]
            
        # Filter low confidence scores
        keep_score_idx = prediction['scores'] > SCORE_THRESH
        b_boxes = prediction['boxes'][keep_score_idx]
        b_scores = prediction['scores'][keep_score_idx]
        b_labels = prediction['labels'][keep_score_idx]

        keep_nms_idx = ops.nms(b_boxes, b_scores, iou_threshold=NMS_IOU_THRESH)
        
        b_boxes = b_boxes[keep_nms_idx].cpu().numpy()
        b_scores = b_scores[keep_nms_idx].cpu().numpy()
        b_labels = b_labels[keep_nms_idx].cpu().numpy()

        detections = []
        for i in range(len(b_boxes)):
            x1, y1, x2, y2 = b_boxes[i]
            detections.append([x1, y1, x2, y2, b_scores[i], b_labels[i]])

        detections = np.array(detections) if len(detections) > 0 else np.empty((0,6))
        tracked_objects = tracker.update(detections)

        # Cleanup state for objects the tracker just dropped (memory leak fix)
        active_ids = set(int(obj[4]) for obj in tracked_objects)
        if hasattr(main, '_prev_ids'):
            for dropped_id in main._prev_ids - active_ids:
                depth_estimator.remove_object(dropped_id)
                ttc_calculator.remove_object(dropped_id)
        main._prev_ids = active_ids

        draw = ImageDraw.Draw(img_pil)
        draw.line(roi_filter.get_polygon_for_drawing(), fill="yellow", width=2)
        
        color_map = {"Danger": "red", "Warning": "orange", "Safe": "lime", "Unknown": "white", "Cross-Traffic": "cyan"}

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id, obj_class = obj
            bbox = [x1, y1, x2, y2]
            
            # Smart 3-zone ROI check (Sync with cws_main.py)
            zone = roi_filter.get_overlap_zone(bbox, obj_class)
            # Depth Estimation with Smoothing (Sync with cws_main.py)
            depth = depth_estimator.estimate_depth(bbox, obj_class, obj_id=obj_id)
            
            if zone == "in_path":
                ttc, alert_level = ttc_calculator.update_and_calculate(obj_id, depth, bbox)
                color = color_map.get(alert_level, "white")
                if alert_level == "Cross-Traffic":
                    text = f"ID:{int(obj_id)} Cls:{int(obj_class)} | Cross-Traffic"
                else:
                    ttc_str = f"{ttc:.1f}s" if ttc != float('inf') else "Inf"
                    text = f"ID:{int(obj_id)} Cls:{int(obj_class)} | Z:{depth:.1f}m | TTC:{ttc_str}"
            elif zone == "margin":
                ttc, alert_level = ttc_calculator.update_and_calculate(obj_id, depth, bbox)
                color = "cyan"
                ttc_str = f"{ttc:.1f}s" if ttc != float('inf') else "Inf"
                text = f"ID:{int(obj_id)} Cls:{int(obj_class)} | Margin | TTC:{ttc_str}"
            else:
                color = "gray"
                text = f"ID:{int(obj_id)} Cls:{int(obj_class)} | Out"
                ttc_calculator.update_and_calculate(obj_id, depth, bbox)
            
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
            bg_width = max(len(text) * 6.5, 40)
            bg_color = color if color != "gray" else "#333333"
            draw.rectangle(((x1, max(0, y1-15)), (x1 + bg_width, max(0, y1))), fill=bg_color)
            draw.text((x1 + 2, max(0, y1-14)), text, fill="black" if color not in ("gray", "cyan") else "white")

        result_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Calculate FPS including display overhead
        cv2.imshow("CWS Live Simulation", result_cv)
        key = cv2.waitKey(1)
        
        loop_end = time.time()
        elapsed = loop_end - loop_start
        total_processing_time += elapsed
        
        fps_live = 1.0 / elapsed if elapsed > 0 else 0
        avg_fps = frame_count / total_processing_time
        
        # Display FPS statistics
        cv2.putText(result_cv, f"Live System FPS: {fps_live:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(result_cv, f"Total Avg FPS: {avg_fps:.1f}", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Re-show with FPS overlays
        cv2.imshow("CWS Live Simulation", result_cv)

        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Video processing finished. Average FPS: {frame_count / total_processing_time:.2f}")

if __name__ == "__main__":
    main()
