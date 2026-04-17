import os
import glob
import torch
import torchvision
import torchvision.ops as ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
import numpy as np

from detectors.base_detector import BaseDetector
from config import BASE_PATH, NUM_CLASSES, SCORE_THRESH, NMS_IOU_THRESH


class FRCNNDetector(BaseDetector):
    def __init__(
        self,
        model_path=None,
    ):
        self.model_path = model_path or self._find_checkpoint()
        if not self.model_path:
            raise FileNotFoundError(
                "No checkpoint found. Train the model first or pass model_path explicitly."
            )

        self.num_classes = NUM_CLASSES
        self.confidence_threshold = SCORE_THRESH
        self.nms_iou_threshold = NMS_IOU_THRESH
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transform = T.Compose([T.ToTensor()])
        self.load_model()

    def _find_checkpoint(self):
        checkpoint_dir = os.path.join(BASE_PATH, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            return None
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_path):
            return best_path
        files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
        return max(files, key=os.path.getmtime) if files else None

    def load_model(self):
        # Build architecture
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # Load checkpoint weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        # Support full checkpoint dict and raw state_dict
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded weights from: {self.model_path}")

        model.to(self.device)
        model.eval()
        self.model = model

    def preprocess(self, image):
        """
        Convert a PIL Image to a normalized Tensor and move it to the target device.

        Args:
            image: PIL Image (RGB).

        Returns:
            torch.Tensor of shape [C, H, W] on self.device.
        """
        return self.transform(image).to(self.device)

    def postprocess(self, raw_output, original_image=None) -> np.ndarray:
        """
        Convert raw Faster R-CNN output dict into standard CWS [N, 6] array.

        Args:
            raw_output (dict): Direct output from the Faster R-CNN model for one image.
                Keys: 'boxes' [M,4], 'scores' [M], 'labels' [M].
            original_image: Unused for FRCNN (coordinates are already in pixel space).

        Returns:
            np.ndarray of shape [N, 6] (absolute pixel coordinates).
            Returns np.empty((0, 6)) when no valid detections remain.
        """
        # Confidence filtering
        keep = raw_output['scores'] > self.confidence_threshold
        boxes  = raw_output['boxes'][keep]
        scores = raw_output['scores'][keep]
        labels = raw_output['labels'][keep]

        if boxes.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)

        # Class-agnostic NMS
        keep_nms = ops.nms(boxes, scores, iou_threshold=self.nms_iou_threshold)
        boxes  = boxes[keep_nms].cpu().numpy()
        scores = scores[keep_nms].cpu().numpy()
        labels = labels[keep_nms].cpu().numpy()

        # Build [N, 6] array
        detections = np.column_stack([boxes, scores, labels]).astype(np.float32)
        return detections

    def detect(self, image) -> np.ndarray:
        """
        Run the full detection pipeline on a single PIL Image.

        Args:
            image: PIL Image (RGB).

        Returns:
            np.ndarray of shape [N, 6]: [x1, y1, x2, y2, score, label]
        """
        img_tensor = self.preprocess(image)

        with torch.no_grad():
            raw_output = self.model([img_tensor])[0]

        return self.postprocess(raw_output, original_image=image)
