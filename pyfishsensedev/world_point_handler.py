from pathlib import Path
from typing import Tuple

import numpy as np

from pyfishsensedev.library.array_read_write import (
    read_camera_calibration,
    read_laser_calibration,
)
from pyfishsensedev.library.constants import PIXEL_PITCH_MM
from pyfishsensedev.library.laser_parallax import (
    compute_world_point,
    compute_world_point_from_depth,
)


class WorldPointHandler:
    @property
    def calibration_matrix(self) -> np.ndarray:
        return self._calibration_matrix

    @calibration_matrix.setter
    def calibration_matrix(self, calibration_matrix: np.ndarray):
        self._calibration_matrix = calibration_matrix

    @property
    def distortion_coeffs(self) -> np.ndarray:
        return self._distortion_coeffs

    @distortion_coeffs.setter
    def distortion_coeffs(self, distortion_coeffs: np.ndarray):
        self._distortion_coeffs = distortion_coeffs

    @property
    def laser_position(self) -> np.ndarray:
        return self._laser_position

    @laser_position.setter
    def laser_position(self, laser_position: np.ndarray):
        self._laser_position = laser_position

    @property
    def laser_orientation(self) -> np.ndarray:
        return self._laser_orientation

    @laser_orientation.setter
    def laser_orientation(self, laser_orientation: np.ndarray):
        self._laser_orientation = laser_orientation

    def __init__(self, lens_calibration_path: Path, laser_calibration_path: Path):
        self._calibration_matrix, self._distortion_coeffs = read_camera_calibration(
            lens_calibration_path.as_posix()
        )
        self._laser_position, self._laser_orientation = read_laser_calibration(
            laser_calibration_path.as_posix()
        )

        focal_length_mm = self.calibration_matrix[0][0] * PIXEL_PITCH_MM
        sensor_size_px = np.array([4000, 3000])
        self.principal_point = self.calibration_matrix[:2, 2]
        self.camera_params = (
            focal_length_mm,
            sensor_size_px[0],
            sensor_size_px[1],
            PIXEL_PITCH_MM,
        )

    def calculate_laser_parallax(self, coord2d: np.ndarray) -> np.ndarray:
        return compute_world_point(
            laser_origin=self.laser_position,
            laser_axis=self.laser_orientation,
            camera_params=self.camera_params,
            image_coordinate=(self.principal_point - coord2d),
        )

    def calculate_world_coordinates_with_depth(
        self, left_coord: np.ndarray, right_coord: np.ndarray, depth: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_coord3d = compute_world_point_from_depth(
            camera_params=self.camera_params,
            image_coordinate=(self.principal_point - left_coord),
            depth=depth,
        )
        right_coord3d = compute_world_point_from_depth(
            camera_params=self.camera_params,
            image_coordinate=(self.principal_point - right_coord),
            depth=depth,
        )

        return left_coord3d, right_coord3d


if __name__ == "__main__":
    world_point_handler = WorldPointHandler(
        Path("./data/lens-calibration.pkg"), Path("./data/laser-calibration.pkg")
    )
