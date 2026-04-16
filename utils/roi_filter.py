import numpy as np
import cv2
from matplotlib.path import Path

class ROIFilter:
    def __init__(self, image_width=1242, image_height=375):
        self.image_width = image_width
        self.image_height = image_height

        y_mid = int(image_height * 0.5)
        y_bot = image_height

        # Per-side default lines stored as polynomial coefficients [a, b] (x = a*y + b)
        self.default_poly_left  = np.polyfit([y_bot, y_mid],
                                             [int(image_width * 0.25), int(image_width * 0.45)], 1)
        self.default_poly_right = np.polyfit([y_bot, y_mid],
                                             [int(image_width * 0.75), int(image_width * 0.55)], 1)

        # Per-side EMA state (start at default)
        self.smooth_poly_left  = self.default_poly_left.copy()
        self.smooth_poly_right = self.default_poly_right.copy()

        # Region of interest for Hough transform
        self.hough_roi = np.array([[
            (0, image_height),
            (int(image_width * 0.4), int(image_height * 0.5)),
            (int(image_width * 0.6), int(image_height * 0.5)),
            (image_width, image_height)
        ]], dtype=np.int32)

        # Temporal smoothing coefficient (higher = slower, more stable)
        self.alpha = 0.8

        # Build initial polygon and path from defaults
        self.polygon_coords = self._build_polygon(self.smooth_poly_left, self.smooth_poly_right)
        self.roi_path = Path(self.polygon_coords)

        # --- Zone thresholds ---
        # overlap_ratio >= IN_PATH_THRESH  -> "in_path"
        # overlap_ratio >= MARGIN_THRESH   -> "margin"
        # otherwise                        -> "out_of_path"
        self.IN_PATH_THRESH = 0.50
        self.MARGIN_THRESH  = 0.20

        # Grid resolution for overlap sampling
        self.GRID_COLS = 5
        self.GRID_ROWS = 5

    def update(self, image):
        # Channel 1: White lane markings 
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 190, 0])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges_all   = cv2.Canny(blur, 50, 150)
        edges_white = cv2.bitwise_and(edges_all, white_mask)

        # Channel 2: General structural edges
        edges_general = cv2.Canny(blur, 80, 200)
        edges_general = cv2.bitwise_and(edges_general, cv2.bitwise_not(white_mask))

        # Mask both channels to hough ROI
        roi_mask = np.zeros_like(edges_all)
        cv2.fillPoly(roi_mask, self.hough_roi, 255)
        masked_white   = cv2.bitwise_and(edges_white,   roi_mask)
        masked_general = cv2.bitwise_and(edges_general, roi_mask)

        # HoughLinesP on each channel
        lines_white   = cv2.HoughLinesP(masked_white,   rho=1, theta=np.pi/180, threshold=20, minLineLength=15, maxLineGap=250)
        lines_general = cv2.HoughLinesP(masked_general, rho=1, theta=np.pi/180, threshold=30, minLineLength=40, maxLineGap=100)

        lw, rw = self._classify_lines(lines_white)
        lg, rg = self._classify_lines(lines_general)

        # Use white if enough white lines, else combine or fall back to general
        left_lines  = lw if len(lw) >= 2 else (lw + lg if lw else lg)
        right_lines = rw if len(rw) >= 2 else (rw + rg if rw else rg)

        poly_left  = self._extrapolate_lines(left_lines)
        poly_right = self._extrapolate_lines(right_lines)

        if poly_left  is None: poly_left  = self.default_poly_left
        if poly_right is None: poly_right = self.default_poly_right

        # EMA on polynomial coefficients
        self.smooth_poly_left  = self.alpha * self.smooth_poly_left  + (1 - self.alpha) * poly_left
        self.smooth_poly_right = self.alpha * self.smooth_poly_right + (1 - self.alpha) * poly_right

        # Build final polygon
        self.polygon_coords = self._build_polygon(self.smooth_poly_left, self.smooth_poly_right)
        self.roi_path = Path(self.polygon_coords)

    def _build_polygon(self, poly_left, poly_right):
        y_bottom    = self.image_height
        y_top_limit = int(self.image_height * 0.5)
        y_top       = y_top_limit

        a_L, b_L = poly_left
        a_R, b_R = poly_right

        # Compute vanishing point so the polygon never crosses beyond it
        if a_L != a_R:
            vp_y = (b_R - b_L) / (a_L - a_R)
            if y_top_limit < vp_y < y_bottom:
                y_top = int(vp_y) + 5  # Stop just before the crossing point

        # Evaluate X at top and bottom, clamped to image width
        def x_at(poly, y):
            return int(np.clip(poly[0] * y + poly[1], 0, self.image_width))

        xl_bot = x_at(poly_left,  y_bottom)
        xl_top = x_at(poly_left,  y_top)
        xr_bot = x_at(poly_right, y_bottom)
        xr_top = x_at(poly_right, y_top)

        # Left edge must be to the left of right edge
        if xl_bot < xr_bot and xl_top < xr_top:
            return [(xl_top, y_top), (xr_top, y_top), (xr_bot, y_bottom), (xl_bot, y_bottom)]

        # Geometry is invalid: return a hardcoded safe default polygon
        return [(int(self.image_width * 0.45), int(self.image_height * 0.5)),
                (int(self.image_width * 0.55), int(self.image_height * 0.5)),
                (int(self.image_width * 0.75), self.image_height),
                (int(self.image_width * 0.25), self.image_height)]

    def _classify_lines(self, raw_lines):
        """Split raw HoughLinesP output into left and right lane candidates."""
        left_lines, right_lines = [], []
        if raw_lines is None:
            return left_lines, right_lines
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.4:  # Filter near-horizontal lines
                continue
            cx = (x1 + x2) / 2
            if slope < 0 and cx < self.image_width * 0.55:   # Left side
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0 and cx > self.image_width * 0.45: # Right side
                right_lines.append((x1, y1, x2, y2))
        return left_lines, right_lines

    def _extrapolate_lines(self, lines):
        """Fit a single line (x = ay + b) through a list of line segments."""
        if not lines:
            return None

        slopes = [(y2 - y1) / (x2 - x1) for x1, y1, x2, y2 in lines]
        med_slope = np.median(slopes)

        # Remove outliers whose slope deviates too far from the median
        refined = [lines[i] for i, s in enumerate(slopes) if abs(s - med_slope) < 0.15]
        if not refined:
            refined = lines  # Keep all if filter is too aggressive

        x_pts, y_pts = [], []
        for x1, y1, x2, y2 in refined:
            x_pts.extend([x1, x2])
            y_pts.extend([y1, y2])

        # Linear regression: x = a*y + b
        return np.polyfit(y_pts, x_pts, 1)

    def _get_sample_points(self, bbox, obj_class):
        """
        Generate a grid of sample points from the bbox region that is most
        relevant to road contact, based on object class.

        Class mapping (KITTI-style):
          1=Car, 2=Van, 3=Truck, 4=Pedestrian, 5=Cyclist

        Strategy:
          - Vehicles (1,2,3): sample bottom 40% of bbox (wheel/ground region)
          - Pedestrian (4):   sample a narrow vertical strip at center X (legs/feet)
          - Cyclist (5):      sample center column, bottom 60%
          - Unknown:          sample bottom 50%
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        cls = int(obj_class)

        if cls in (1, 2, 3):
            # Vehicles: only bottom 40% of height — wheel contact zone
            sample_y1 = y1 + h * 0.60
            sample_y2 = y2
            sample_x1 = x1
            sample_x2 = x2
        elif cls == 4:
            # Pedestrian: narrow vertical strip at center (person is thin)
            cx = (x1 + x2) / 2.0
            strip_half = max(w * 0.15, 8)  # At least 8px wide
            sample_x1 = cx - strip_half
            sample_x2 = cx + strip_half
            sample_y1 = y1 + h * 0.50     # Bottom half (legs/feet)
            sample_y2 = y2
        elif cls == 5:
            # Cyclist: center column, bottom 60%
            cx = (x1 + x2) / 2.0
            strip_half = max(w * 0.20, 10)
            sample_x1 = cx - strip_half
            sample_x2 = cx + strip_half
            sample_y1 = y1 + h * 0.40
            sample_y2 = y2
        else:
            # Unknown: bottom half
            sample_x1 = x1
            sample_x2 = x2
            sample_y1 = y1 + h * 0.50
            sample_y2 = y2

        # Clamp to image bounds
        sample_x1 = max(sample_x1, 0)
        sample_x2 = min(sample_x2, self.image_width)
        sample_y1 = max(sample_y1, 0)
        sample_y2 = min(sample_y2, self.image_height - 1)

        # Degenerate box guard
        if sample_x2 <= sample_x1 or sample_y2 <= sample_y1:
            cx = (x1 + x2) / 2.0
            return [(cx, min(y2, self.image_height - 1))]

        # Build uniform grid of GRID_COLS x GRID_ROWS sample points
        xs = np.linspace(sample_x1, sample_x2, self.GRID_COLS)
        ys = np.linspace(sample_y1, sample_y2, self.GRID_ROWS)
        points = [(float(x), float(y)) for y in ys for x in xs]
        return points

    def get_overlap_ratio(self, bbox, obj_class=0):
        """
        Compute the fraction of class-aware sample points inside the lane polygon.
        Returns a float in [0.0, 1.0].
        """
        points = self._get_sample_points(bbox, obj_class)
        if not points:
            return 0.0
        inside = sum(1 for pt in points if self.roi_path.contains_point(pt))
        return inside / len(points)

    def get_overlap_zone(self, bbox, obj_class=0):
        """
        Classify a bounding box into one of three zones based on overlap ratio:
          'in_path'     — object is mainly inside the lane (>= 50%)
          'margin'      — object partially overlaps the lane (20-50%)
          'out_of_path' — object is outside the lane (< 20%)
        """
        ratio = self.get_overlap_ratio(bbox, obj_class)
        if ratio >= self.IN_PATH_THRESH:
            return "in_path"
        elif ratio >= self.MARGIN_THRESH:
            return "margin"
        else:
            return "out_of_path"

    def is_in_path(self, bbox):
        """
        Backward-compatible check. Returns True if zone is 'in_path' or 'margin'.
        Prefer using get_overlap_zone() directly for finer control.
        """
        zone = self.get_overlap_zone(bbox, obj_class=0)
        return zone in ("in_path", "margin")

    def get_polygon_for_drawing(self):
        return self.polygon_coords + [self.polygon_coords[0]]
