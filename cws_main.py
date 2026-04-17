import os
import glob
from PIL import Image, ImageDraw
import numpy as np
import cv2
from utils.calib import Calibration
from utils.sort import Sort
from utils.depth_estimation import DepthEstimator
from utils.ttc import TTCCalculator
from utils.roi_filter import ROIFilter
from detectors.frcnn_detector import FRCNNDetector
from config import SEQ_PATH, CALIB_FILE, OUTPUT_PATH

def cleanup_output_dir():
    if os.path.exists(OUTPUT_PATH):
        for f in os.listdir(OUTPUT_PATH):
            os.remove(os.path.join(OUTPUT_PATH, f))
    else:
        os.makedirs(OUTPUT_PATH)

def main():
    cleanup_output_dir()

    try:
        detector = FRCNNDetector()
    except FileNotFoundError as e:
        print(e)
        return
    print(f"Using device: {detector.device}")

    # Init CWS modules
    calib = Calibration(CALIB_FILE)
    focal_length = calib.get_focal_length_x(camera_id=2)
    print(f"Camera 02 Focal length (f_x): {focal_length}")

    tracker = Sort(max_age=1, min_hits=1, iou_threshold=0.35)
    depth_estimator = DepthEstimator(focal_length_x=focal_length)
    ttc_calculator = TTCCalculator(fps=10.0)

    roi_filter = ROIFilter(image_width=1242, image_height=375)

    image_files = sorted(glob.glob(os.path.join(SEQ_PATH, "*.png")))
    if not image_files:
        print(f"Image sequence not found at {SEQ_PATH}")
        return

    print("Starting CWS processing pipeline...")
    for idx, img_path in enumerate(image_files):
        img_pil = Image.open(img_path).convert("RGB")

        # Update dynamic ROI based on Lane Detection
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        roi_filter.update(img_cv)

        # Single unified call — works for any detector subclass
        prediction = detector.detect(img_pil)
            
        # prediction is already [N, 6]: [x1, y1, x2, y2, score, label]
        tracked_objects = tracker.update(prediction)

        # Cleanup state for objects the tracker just dropped (memory leak fix)
        active_ids = set(int(obj[4]) for obj in tracked_objects)
        if hasattr(main, '_prev_ids'):
            for dropped_id in main._prev_ids - active_ids:
                depth_estimator.remove_object(dropped_id)
                ttc_calculator.remove_object(dropped_id)
        main._prev_ids = active_ids

        draw = ImageDraw.Draw(img_pil)
        draw.line(roi_filter.get_polygon_for_drawing(), fill="yellow", width=2)
        
        # Color map for TTC alert levels
        color_map = {"Danger": "red", "Warning": "orange", "Safe": "lime", "Unknown": "white", "Cross-Traffic": "cyan"}

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id, obj_class = obj
            bbox = [x1, y1, x2, y2]

            # --- Smart 3-zone ROI check (class-aware grid overlap) ---
            zone = roi_filter.get_overlap_zone(bbox, obj_class)
            depth = depth_estimator.estimate_depth(bbox, obj_class, obj_id=obj_id)

            if zone == "in_path":
                # Fully in lane: normal TTC evaluation
                ttc, alert_level = ttc_calculator.update_and_calculate(obj_id, depth, bbox)
                color = color_map.get(alert_level, "white")

                if alert_level == "Cross-Traffic":
                    ttc_str = "Inf"
                    text = f"ID:{int(obj_id)} Cls:{int(obj_class)} | Cross-Traffic"
                else:
                    ttc_str = f"{ttc:.1f}s" if ttc != float('inf') else "Inf"
                    text = f"ID:{int(obj_id)} Cls:{int(obj_class)} | Z:{depth:.1f}m | TTC:{ttc_str}"

            elif zone == "margin":
                # Partially overlapping lane edge: compute TTC but flag as marginal
                ttc, alert_level = ttc_calculator.update_and_calculate(obj_id, depth, bbox)
                color = "cyan"

                if alert_level == "Cross-Traffic":
                    text = f"ID:{int(obj_id)} Cls:{int(obj_class)} | Margin | Cross-Traffic"
                else:
                    ttc_str = f"{ttc:.1f}s" if ttc != float('inf') else "Inf"
                    text = f"ID:{int(obj_id)} Cls:{int(obj_class)} | Margin Z:{depth:.1f}m TTC:{ttc_str}"

            else:
                # Fully outside lane: skip TTC, still track for history continuity
                color = "gray"
                text = f"ID:{int(obj_id)} Cls:{int(obj_class)} | Out of Path"
                ttc_calculator.update_and_calculate(obj_id, depth, bbox)

            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)

            bg_width = max(len(text) * 6, 40)
            bg_color  = color if color != "gray" else "#333333"
            text_fill = "black" if color not in ("gray", "white", "cyan") else ("white" if color in ("gray", "cyan") else "black")
            draw.rectangle(((x1, max(0, y1 - 15)), (x1 + bg_width, max(0, y1))), fill=bg_color)
            draw.text((x1 + 2, max(0, y1 - 14)), text, fill=text_fill)

        output_file = os.path.join(OUTPUT_PATH, os.path.basename(img_path))
        img_pil.save(output_file)
        
        if (idx + 1) % 20 == 0:
            print(f"Processed frame {idx+1}/{len(image_files)}")

    print(f"DONE! {len(image_files)} frames have been saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
