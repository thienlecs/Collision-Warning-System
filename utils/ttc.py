import numpy as np

class TTCCalculator:
    def __init__(self, fps=10.0, alpha=0.3):
        self.fps = fps
        self.alpha = alpha # EMA smoothing factor
        self.history = {} # Store depth Z
        self.x_history = {} # Store center X
        
    def update_and_calculate(self, obj_id, current_depth, bbox):
        if current_depth < 0:
            return float('inf'), "Unknown"
            
        center_x = (bbox[0] + bbox[2]) / 2.0
            
        if obj_id not in self.history:
            self.history[obj_id] = []
            self.x_history[obj_id] = []
            smoothed_depth = current_depth
        else:
            # Low-Pass Filter (EMA) to smooth out raw depth noise
            prev_smoothed = self.history[obj_id][-1]
            smoothed_depth = (self.alpha * current_depth) + ((1.0 - self.alpha) * prev_smoothed)
            
        self.history[obj_id].append(smoothed_depth)
        self.x_history[obj_id].append(center_x)
        
        # Tune lower length for more aggressive detection, higher for more conservative
        if len(self.history[obj_id]) > 10:
            self.history[obj_id].pop(0)
            self.x_history[obj_id].pop(0)
            
        if len(self.history[obj_id]) < 2:
            return float('inf'), "Safe"
            
        dt = 1.0 / self.fps
        
        # Weighted Linear Regression over the history buffer
        # Recent frames have higher weights to react faster to sudden changes
        n = len(self.history[obj_id])
        x_time = np.array([i * dt for i in range(n)])
        y_depth = np.array(self.history[obj_id])
        
        # Create weights: e.g., if n=5, weights = [1, 2, 3, 4, 5]
        # This makes the most recent frame 5 times more important than the oldest one
        weights = np.linspace(1, 5, n) 
        
        # Fit line: y = m*x + c with weights
        m_depth, c_depth = np.polyfit(x_time, y_depth, 1, w=weights)
        
        # If object is approaching, depth is decreasing over time, so slope (m_depth) is negative.
        v_rel = -m_depth
        
        # Linear Regression for Lateral Velocity (also weighted)
        y_x_pos = np.array(self.x_history[obj_id])
        m_x, _ = np.polyfit(x_time, y_x_pos, 1, w=weights)
        v_x_pixels = abs(m_x)
        
        if v_x_pixels > 150:
            return float('inf'), "Cross-Traffic"
        
        # If v_rel > 0 means the object is approaching
        if v_rel > 0:
            ttc = smoothed_depth / v_rel
        else:
            ttc = float('inf') 
            
        if ttc < 2.5:
            warning_level = "Danger"
        elif ttc < 5.0:
            warning_level = "Warning"
        else:
            warning_level = "Safe"
            
        return ttc, warning_level

    def remove_object(self, obj_id):
        """Clean up history for objects that are no longer tracked.
        Call this whenever the tracker drops an object to prevent unbounded
        memory growth in long video sequences.
        """
        self.history.pop(obj_id, None)
        self.x_history.pop(obj_id, None)
