from abc import ABC
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from e4e_camera_calibration.cameras.camera import Camera


class Calibrator(ABC):
    def __init__(self, camera: Camera):
        self._camera = camera

    def generate_calibration_images(
        self, output_dir: str, prefix: str, rows=14, columns=10
    ) -> bool:
        found_calibration_images: int = 0
        idx = 0
        prev_idx = idx
        with tqdm(total=self._camera.image_count) as progress:
            for image in self._camera:
                if self._is_valid_calibration_image(image, rows, columns):
                    self._camera._write_image_file(
                        image, Path(output_dir, f"{prefix}{idx}")
                    )
                    idx = self._search_next_different_image(
                        image, idx, self._camera.image_count - 1
                    )
                    self._camera.seek(idx)
                    found_calibration_images = found_calibration_images + 1
                else:
                    idx = self._search_next_calibration_image(
                        idx, self._camera.image_count - 1, rows, columns
                    )
                    self._camera.seek(idx)

                progress.update(idx - prev_idx)

                if self._camera.image_count - 1 - idx <= 1:
                    break

                prev_idx = idx

        if found_calibration_images > 0:
            with open(
                Path(output_dir, f"{prefix}metadata.txt"), "w", encoding="utf8"
            ) as file:
                file.write(f"{found_calibration_images}\n")

        return found_calibration_images > 0

    def _calculate_difference_score(self, image1: np.ndarray, image2: np.ndarray):
        return ssim(image1, image2, channel_axis=2)

    def _get_chessboard_corners(self, image: np.ndarray, rows: int, columns: int):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret:

            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            # cv2.drawChessboardCorners(frame, (rows,columns), corners, ret)
            # cv2.imshow('img', frame)
            # k = cv2.waitKey(500)

            return corners
        else:
            return None

    def _is_valid_calibration_image(
        self, image: np.ndarray, rows: int, columns: int, min_percent_size=0
    ):
        corners = self._get_chessboard_corners(image, rows, columns)

        if corners is None:
            return False

        img_height, img_width, _ = image.shape
        img_area = img_height * img_width

        top_left = corners[0, 0, :]
        bottom_right = corners[-1, 0, :]

        height = bottom_right[0] - top_left[0]
        width = bottom_right[1] - top_left[1]
        area = height * width

        return area / img_area >= min_percent_size

    def _search_next_calibration_image(
        self,
        min_idx: int,
        max_idx: int,
        rows: int,
        columns: int,
        direction=1,
        _start_idx=None,
        _prev_jump=None,
        _first_pass=True,
    ):
        if direction == 1:
            _start_idx = _start_idx or min_idx
        elif direction == -1:
            _start_idx = _start_idx or max_idx
        else:
            raise ValueError(
                f"direction {direction} is not valid.  Please use 1 or -1."
            )

        n1, n2 = _prev_jump or (0, 1)
        jump = n1 + n2

        if direction == 1:
            next_idx = min_idx + jump
        elif direction == -1:
            next_idx = max_idx - jump

        next_idx = min(next_idx, max_idx)
        next_idx = max(min_idx, next_idx)

        if next_idx >= max_idx:
            return max_idx

        if self._is_valid_calibration_image(self._camera.at(next_idx), rows, columns):
            if not _first_pass:
                return next_idx

            if direction == 1:
                return self._search_next_calibration_image(
                    _start_idx, next_idx, rows, columns, direction=-1
                )
            elif direction == -1:
                return self._search_next_calibration_image(
                    min_idx,
                    next_idx,
                    rows,
                    columns,
                    direction=-1,
                    _start_idx=_start_idx,
                    _prev_jump=(n2, jump),
                )
        else:
            if direction == 1:
                return self._search_next_calibration_image(
                    next_idx,
                    max_idx,
                    rows,
                    columns,
                    direction=1,
                    _first_pass=_first_pass,
                    _start_idx=_start_idx,
                    _prev_jump=(n2, jump),
                )
            elif direction == -1:
                return self._search_next_calibration_image(
                    next_idx, _start_idx, rows, columns, direction=1, _first_pass=False
                )

    def _search_next_different_image(
        self, image: np.ndarray, min_idx: int, max_idx: int, target_score=0.66
    ):
        delta = max_idx - min_idx

        if delta == 1:
            return max_idx

        mid_idx = min_idx + int(delta / 2)

        score = self._calculate_difference_score(image, self._camera.at(mid_idx))
        if target_score - 0.01 <= score <= target_score + 0.01:
            return mid_idx
        elif score < target_score - 0.01:
            return self._search_next_different_image(image, min_idx, mid_idx)
        elif score > target_score + 0.01:
            return self._search_next_different_image(image, mid_idx, max_idx)
