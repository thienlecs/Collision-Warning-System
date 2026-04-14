class DepthEstimator:
    def __init__(self, focal_length_x):
        self.focal_length_x = focal_length_x
        # Estimate height of object
        # KITTI: 1=Vehicle, 2=Pedestrian, 3=Cyclist
        self.heights = {
            1.0: 1.5,  # Vehicle: 1.5m
            2.0: 1.7,  # Pedestrian: 1.7m
            3.0: 1.6   # Cyclist: 1.6m
        }
        
    def estimate_depth(self, bbox, class_id):
        # bbox format: [x1, y1, x2, y2]
        pixel_height = bbox[3] - bbox[1]
        
        real_height = self.heights.get(float(class_id), 1.5) # default height = 1.5
        
        # Pinhole Camera Model
        # z = (f * real_height) / pixel_height
        if pixel_height > 0:
            depth = (self.focal_length_x * real_height) / pixel_height
        else:
            depth = -1.0 # Error
            
        return depth
