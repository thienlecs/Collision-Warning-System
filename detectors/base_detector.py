import numpy as np

class BaseDetector:
    def load_model(self, *args, **kwargs):
        """Load the model weights into self.model."""
        raise NotImplementedError

    def preprocess(self, *args, **kwargs):
        """Prepare the raw input image for the model."""
        raise NotImplementedError

    def postprocess(self, *args, **kwargs) -> np.ndarray:
        """Convert raw model output into the standard CWS detection format [N, 6]."""
        raise NotImplementedError

    def detect(self, *args, **kwargs) -> np.ndarray:
        """
        The detect() method must always return a NumPy array of shape [N, 6]:
        [x1, y1, x2, y2, score, label]
        where coordinates are absolute pixel values (not normalized).
        
        Pipeline in this function: preprocess -> model prediction -> postprocess."""
        raise NotImplementedError
