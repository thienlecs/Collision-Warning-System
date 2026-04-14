class TTCCalculator:
    def __init__(self, fps=10.0):
        self.fps = fps
        self.history = {} # Store depth Z
        self.x_history = {} # Store center X
        
    def update_and_calculate(self, obj_id, current_depth, bbox):
        if current_depth < 0:
            return float('inf'), "Unknown"
            
        center_x = (bbox[0] + bbox[2]) / 2.0
            
        if obj_id not in self.history:
            self.history[obj_id] = []
            self.x_history[obj_id] = []
            
        self.history[obj_id].append(current_depth)
        self.x_history[obj_id].append(center_x)
        
        if len(self.history[obj_id]) > 5:
            self.history[obj_id].pop(0)
            self.x_history[obj_id].pop(0)
            
        if len(self.history[obj_id]) < 2:
            return float('inf'), "Safe"
            
        dt = 1.0 / self.fps
        
        # Use simple smoothing by calculating velocity over the entire history window
        # instead of just comparing it to the immediate previous frame.
        time_elapsed = dt * (len(self.history[obj_id]) - 1)

        depth_old = self.history[obj_id][0]
        v_rel = (depth_old - current_depth) / time_elapsed
        
        # Lateral Velocity based on Pixel position
        x_old = self.x_history[obj_id][0]
        v_x_pixels = abs(center_x - x_old) / time_elapsed
        
        # If lateral velocity > 250 pixels/second (about 1/4 of the image per second), the object is turning
        if v_x_pixels > 250:
            return float('inf'), "Cross-Traffic"
        
        # If v_rel > 0 means the object is approaching
        if v_rel > 0:
            ttc = current_depth / v_rel
        else:
            ttc = float('inf') 
            
        if ttc < 2.5:
            warning_level = "Danger"
        elif ttc < 5.0:
            warning_level = "Warning"
        else:
            warning_level = "Safe"
            
        return ttc, warning_level
