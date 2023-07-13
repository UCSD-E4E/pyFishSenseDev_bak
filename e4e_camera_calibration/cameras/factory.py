from e4e_camera_calibration.cameras.camera import Camera
from e4e_camera_calibration.cameras.qoocam import QoocamEgoStereoCamera
from e4e_camera_calibration.cameras.olympus import OlympusTG6MonoCamera


CAMERA_MAP = {"qoocam-ego": QoocamEgoStereoCamera, "Olympus-TG6": OlympusTG6MonoCamera}


def str2camera(camera_name: str, **kwargs) -> Camera:
    if camera_name in CAMERA_MAP:
        return CAMERA_MAP[camera_name](**kwargs)

    raise ValueError(f"{camera_name} is not supported.")
