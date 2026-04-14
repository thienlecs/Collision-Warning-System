import numpy as np
import os

class Calibration:
    def __init__(self, calib_file):
        self.calib_file = calib_file
        self.P0, self.P1, self.P2, self.P3 = self.read_calib_file()
        
    def read_calib_file(self):
        with open(self.calib_file, 'r') as f:
            lines = f.readlines()
            
        P0 = P1 = P2 = P3 = None
        for line in lines:
            if line.startswith('P_rect_00:'):
                P0 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            elif line.startswith('P_rect_01:'):
                P1 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            elif line.startswith('P_rect_02:'):
                P2 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            elif line.startswith('P_rect_03:'):
                P3 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
                
        return P0, P1, P2, P3

    def get_focal_length_x(self, camera_id=2):
        if camera_id == 2:
            return self.P2[0, 0]
        elif camera_id == 3:
            return self.P3[0, 0]
        return None
