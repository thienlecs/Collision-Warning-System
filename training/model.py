import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from training.config import NUM_CLASSES

def get_model(num_classes=NUM_CLASSES, device=None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # Replace the box predictor with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if device:
        model.to(device)
    return model
