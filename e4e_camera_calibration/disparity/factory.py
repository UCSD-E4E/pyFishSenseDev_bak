from e4e_camera_calibration.disparity.disparity import SGBMDisparity
from e4e_camera_calibration.cameras.calibrated_stereo_camera import (
    CalibratedStereoCamera,
)

DISPARITY_MAP = {"SGBM": SGBMDisparity}


def str2disparity(
    algorithm_name: str, calibrated_camera: CalibratedStereoCamera, **kwargs
):
    if algorithm_name in DISPARITY_MAP:
        return DISPARITY_MAP[algorithm_name](calibrated_camera, **kwargs)

    raise ValueError(f"{algorithm_name} is not supported.")
