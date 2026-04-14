import numpy as np
from matplotlib.path import Path

class ROIFilter:
    def __init__(self, image_width=1242, image_height=375):

        top_left = (int(image_width * 0.45), int(image_height * 0.5)) 
        top_right = (int(image_width * 0.55), int(image_height * 0.5))
        bottom_right = (int(image_width * 0.75), image_height)
        bottom_left = (int(image_width * 0.25), image_height)
        
        self.polygon_coords = [top_left, top_right, bottom_right, bottom_left]
        self.roi_path = Path(self.polygon_coords)
        self.image_height = image_height

    def is_in_path(self, bbox):
        # Position of the contact point between the object and the road
        center_x = (bbox[0] + bbox[2]) / 2.0
        bottom_y = min(bbox[3], self.image_height - 1) # Force point to stay inside vertical bounds
        
        # True if inside the lane
        return self.roi_path.contains_point((center_x, bottom_y))
        
    def get_polygon_for_drawing(self):
        return self.polygon_coords + [self.polygon_coords[0]]
