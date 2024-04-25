from pathlib import Path

import cv2
import numpy as np

from pyfishsensedev.library.array_read_write import read_camera_calibration


class ImageRectifier:
    def __init__(self, lens_calibration_path: Path):
        self.calibration_matrix, self.distortion_coeffs = read_camera_calibration(
            lens_calibration_path.as_posix()
        )

    def rectify(self, img: np.ndarray) -> np.ndarray:
        return cv2.undistort(img, self.calibration_matrix, self.distortion_coeffs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = cv2.imread("./data/png/P7170081.png")

    image_rectifier = ImageRectifier(Path("./data/lens-calibration.pkg"))
    undist_img = image_rectifier.rectify(img)

    plt.imshow(undist_img)
    plt.show()
