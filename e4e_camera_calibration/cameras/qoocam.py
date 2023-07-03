from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from e4e_camera_calibration.cameras.stereo_camera import StereoCamera


class QoocamEgoStereoCamera(StereoCamera):
    def __init__(self, **kwargs) -> None:
        super().__init__(manufacturer="Qoocam", model="Ego", **kwargs)

        self._capture: cv2.VideoCapture = None

    def _load_image_file(self, file_path: str):
        return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

    def _load_video(self, file_path: str):
        self._capture = cv2.VideoCapture(file_path)
        self._image_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def _preprocess_image(self, image: np.ndarray):
        _, width, _ = image.shape

        left = image[:, : int(width / 2), :]
        right = image[:, int(width / 2) :, :]

        return left, right

    def _process_directory(self, file_path: str):
        files = [str(p) for p in Path(file_path).iterdir() if p.is_file()]
        files.sort()
        return files

    def _next_video_frame(self):
        success, frame = self._capture.read()

        if success:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return None

    def _seek_video_file(self, index: int):
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)

    def _write_image_file(self, image: Tuple[np.ndarray, np.ndarray], file_path: Path):
        file_path = file_path.as_posix()
        file_path = f"{file_path}.png"
        left, right = image
        height, width, channels = left.shape

        image = np.zeros((height, width * 2, channels), dtype=np.uint8)
        image[:, :width, :] = left
        image[:, width:, :] = right

        cv2.imwrite(f"{file_path}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
