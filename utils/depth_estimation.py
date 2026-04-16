class DepthEstimator:
    def __init__(self, focal_length_x, ph_alpha=0.4, max_depth_jump=5.0):
        self.focal_length_x = focal_length_x
        # Estimate height of object
        # KITTI: 1=Car, 2=Van, 3=Truck, 4=Pedestrian, 5=Cyclist
        self.heights = {
            1.0: 1.5,  # Car: 1.5m
            2.0: 1.8,  # Van: 1.8m
            3.0: 2.0,  # Truck: 2.0m
            4.0: 1.7,  # Pedestrian: 1.7m
            5.0: 1.6   # Cyclist: 1.6m
        }
        # EMA smoothing factor for pixel_height (lower = smoother, less reactive)
        # Smoothing pixel_height before division prevents nonlinear noise amplification
        self.ph_alpha = ph_alpha

        # Maximum allowed depth change (meters) between consecutive frames per object.
        # Physically, no tracked object can teleport >5m in a single frame at 10fps.
        self.max_depth_jump = max_depth_jump

        # Per-object state: {obj_id: {smoothed_ph, prev_depth}}
        self._state = {}

    def estimate_depth(self, bbox, class_id, obj_id=None):
        """Estimate depth with optional per-object pixel-height EMA and depth-jump clamping.

        Args:
            bbox: [x1, y1, x2, y2]
            class_id: KITTI class label
            obj_id: tracker ID used for per-object smoothing. If None, no smoothing applied.
        """
        raw_ph = bbox[3] - bbox[1]  # raw pixel height
        real_height = self.heights.get(float(class_id), 1.5)

        if raw_ph <= 0:
            return -1.0  # Error

        if obj_id is None:
            # No tracking context — compute depth directly
            return (self.focal_length_x * real_height) / raw_ph

        # Smooth pixel_height with EMA
        # This avoids the 1/x nonlinearity amplifying bbox jitter for far objects.
        state = self._state.setdefault(obj_id, {"smoothed_ph": None, "prev_depth": None})

        if state["smoothed_ph"] is None:
            smoothed_ph = raw_ph
        else:
            # Far objects (small bbox) → use lower alpha for stronger smoothing
            # Near objects (large bbox) → alpha stays close to ph_alpha
            distance_factor = min(1.0, raw_ph / 60.0)  # ~1.0 when ph>=60px, smaller when far
            effective_alpha = self.ph_alpha * distance_factor + 0.15 * (1.0 - distance_factor)
            smoothed_ph = effective_alpha * raw_ph + (1.0 - effective_alpha) * state["smoothed_ph"]

        state["smoothed_ph"] = smoothed_ph

        # Compute depth from smoothed pixel_height
        depth = (self.focal_length_x * real_height) / smoothed_ph

        # Clamp per-frame depth jump

        # Prevents a single noisy detection from causing a large discontinuity.
        if state["prev_depth"] is not None:
            delta = depth - state["prev_depth"]
            if abs(delta) > self.max_depth_jump:
                depth = state["prev_depth"] + self.max_depth_jump * (1 if delta > 0 else -1)

        state["prev_depth"] = depth
        return depth

    def remove_object(self, obj_id):
        """Clean up state for objects that are no longer tracked."""
        self._state.pop(obj_id, None)
