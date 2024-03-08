from pathlib import Path
from typing import Tuple

import numpy as np

from pyfishsense.library.array_read_write import read_camera_calibration, read_laser_calibration
from pyfishsense.library.constants import PIXEL_PITCH_MM
from pyfishsense.library.laser_parallax import compute_world_point, compute_world_point_from_depth

class WorldPointHandler:
    def __init__(self, lens_calibration_path: Path, laser_calibration_path: Path):
        self.calibration_matrix, self.distortion_coeffs = read_camera_calibration(lens_calibration_path.as_posix())
        self.laser_position, self.laser_orientation = read_laser_calibration(laser_calibration_path.as_posix())

        focal_length_mm = self.calibration_matrix[0][0] * PIXEL_PITCH_MM
        sensor_size_px = np.array([4000,3000])
        self.principal_point = self.calibration_matrix[:2,2]
        self.camera_params = (focal_length_mm, sensor_size_px[0], sensor_size_px[1], PIXEL_PITCH_MM)

    def calculate_laser_parallax(self, coord2d: np.ndarray) -> np.ndarray:
        return compute_world_point(
            laser_origin=self.laser_position, 
            laser_axis=self.laser_orientation,
            camera_params=self.camera_params, 
            image_coordinate=(self.principal_point - coord2d)
        )
    
    def calculate_world_coordinates_with_depth(self, left_coord: np.ndarray, right_coord: np.ndarray, depth: float) -> Tuple[np.ndarray, np.ndarray]:
        left_coord3d = compute_world_point_from_depth(
            camera_params=self.camera_params, 
            image_coordinate=(self.principal_point - left_coord), 
            depth=depth
        )
        right_coord3d = compute_world_point_from_depth(
            camera_params=self.camera_params, 
            image_coordinate=(self.principal_point - right_coord), 
            depth=depth
        )

        return left_coord3d, right_coord3d

if __name__ == "__main__":
    world_point_handler = WorldPointHandler(Path("./data/lens-calibration.pkg"), Path("./data/laser-calibration.pkg"))