from e4e_camera_calibration.cameras.camera import Camera
from e4e_camera_calibration.cameras.qoocam import QoocamEgoStereoCamera


CAMERA_MAP = {"qoocam-ego": QoocamEgoStereoCamera}


def str2camera(camera_name: str, **kwargs) -> Camera:
    if camera_name in CAMERA_MAP:
        return CAMERA_MAP[camera_name](**kwargs)

    raise ValueError(f"{camera_name} is not supported.")
