from typing import Tuple

import numpy as np

from e4e_camera_calibration.calibrators.calibrator import Calibrator
from e4e_camera_calibration.cameras.stereo_camera import StereoCamera


class StereoCalibrator(Calibrator):
    def __init__(self, camera: StereoCamera):
        super().__init__(camera)
        self._camera = camera

    def _calculate_difference_score(
        self,
        image1: Tuple[np.ndarray, np.ndarray],
        image2: Tuple[np.ndarray, np.ndarray],
    ):
        left1, right1 = image1
        left2, right2 = image2

        return super()._calculate_difference_score(left1, left2)

    def _is_valid_calibration_image(
        self, image: Tuple[np.ndarray, np.ndarray], rows: int, columns: int
    ):
        left, right = image

        return super()._is_valid_calibration_image(
            left, rows, columns
        ) and super()._is_valid_calibration_image(right, rows, columns)
