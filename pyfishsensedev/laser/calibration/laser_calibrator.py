from typing import List

import numpy as np

from pyfishsensedev.laser.calibration.plane_detector import PlaneDetector


class LaserCalibrator:
    def __init__(self, plane_detector: PlaneDetector) -> None:
        self._plane_detector = plane_detector

    def add_image(self, img: np.ndarray, laser_coord: np.ndarray):
        board_plane_points, normal_vector = self._plane_detector.detect(img)


if __name__ == "__main__":
    from glob import glob
    from pathlib import Path

    import torch

    from pyfishsensedev.image.image_processors.raw_processor import RawProcessor
    from pyfishsensedev.image.image_rectifier import ImageRectifier
    from pyfishsensedev.laser.calibration.laser_calibration import LaserCalibration
    from pyfishsensedev.laser.calibration.slate_plane_detector import SlatePlaneDetector
    from pyfishsensedev.laser.laser_detector import LaserDetector

    def uint16_2_double(img: np.ndarray) -> np.ndarray:
        return img.astype(np.float64) / 65535

    def uint16_2_uint8(img: np.ndarray) -> np.ndarray:
        return (uint16_2_double(img) * 255).astype(np.uint8)

    img_paths = glob("/home/chris/data/fishsense/H slate dive 3/*.ORF")
    lens_calibration_path = Path("./data/lens-calibration.pkg")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_processor_hist_eq = RawProcessor()
    raw_processor = RawProcessor(enable_histogram_equalization=False)

    image_rectifier = ImageRectifier(Path("./data/lens-calibration.pkg"))

    laser_detector = LaserDetector(
        lens_calibration_path, LaserCalibration(None), device
    )

    calibrator = LaserCalibrator(
        SlatePlaneDetector(
            Path("./data/Dive Slate#1.pdf").absolute(), lens_calibration_path
        )
    )

    for img_path in img_paths:
        img = raw_processor_hist_eq.load_and_process(Path(img_path))
        img_dark = raw_processor.load_and_process(Path(img_path))

        img = uint16_2_uint8(image_rectifier.rectify(img))
        img_dark = uint16_2_uint8(image_rectifier.rectify(img_dark))

        laser_coord = laser_detector.find_laser(img_dark)

        if laser_coord is None:
            continue

        print(laser_coord)

        calibrator.add_image(img, laser_coord)

    # laser_calibration = calibrator.calibrate()

    # print(laser_calibration)
