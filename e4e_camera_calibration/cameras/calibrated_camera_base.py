from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple
import io
import json
import tarfile

import cv2
import numpy as np

from e4e_camera_calibration.cameras.camera import Camera
from e4e_camera_calibration.cameras.stereo_camera import StereoCamera


class CalibratedCameraBase(Camera, ABC):
    def __init__(self, camera: Camera, **kwargs) -> None:
        super().__init__(**kwargs)

        self._camera = camera

    @property
    def manufacturer(self):
        return self._camera.manufacturer

    @property
    def model(self):
        return self._camera.model

    @property
    def name(self):
        return self._camera.name

    @property
    def number_of_sensors(self) -> int:
        return self._camera.number_of_sensors

    @property
    def serial_number(self):
        return self._camera.serial_number

    @abstractmethod
    def calibrate(
        self,
        rows=14,
        columns=10,
        square_size=4.1 * 10**-3,
        display_calibration_error=False,
        max_error=0.5,
    ):
        raise NotImplementedError()

    def _calibrate_camera(
        self,
        images: Iterable[np.ndarray],
        rows: int,
        columns: int,
        square_size: float,
        max_error=0.5,
        camera_name: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        images = list(images)

        # The following is adapted from https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # coordinates of squares in the checkerboard world space
        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp = square_size * objp

        # frame dimensions. Frames should be the same size.
        width = images[0].shape[1]
        height = images[0].shape[0]

        # Pixel coordinates of checkerboards
        imgpoints = []  # 2d points in image plane.

        # coordinates of the checkerboard in checkerboard world space.
        objpoints = []  # 3d point in real world space

        for frame in images:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

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

                objpoints.append(objp)
                imgpoints.append(corners)

        (
            error,
            camera_matrix,
            distortion_coefficients,
            _,
            _,
        ) = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)

        if error > max_error:
            raise ValueError(
                f"While calibrating {camera_name}, an error of {error} was above the maximum allowed error of {max_error}."
            )

        return error, camera_matrix, distortion_coefficients

    def load(self, path: str):
        return self._camera.load(path)

    def load_calibration(self, file_path: str):
        with tarfile.open(file_path, "r:gz") as f:
            self._load_calibration(f)

    def save_calibration(self, file_path: str):
        metadata = {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "calibrated_date": str(datetime.now()),
        }
        self._add_metadata(metadata)

        path = Path(file_path)
        if path.exists():
            path.unlink()

        with tarfile.open(file_path, "x:gz") as f:
            with io.StringIO() as s:
                json.dump(metadata, s, indent=True)
                bytes = s.getvalue().encode("utf8")
                with io.BytesIO(bytes) as b:
                    tarinfo = tarfile.TarInfo("metadata.json")
                    tarinfo.size = len(bytes)
                    tarinfo.mtime = int(datetime.now().timestamp())
                    f.addfile(tarinfo, b)

            self._save_calibration(f)

    def _add_metadata(self, metadata: Dict[str, any]):
        pass

    def _add_numpy_array(
        self, array: np.ndarray, file_name: str, file: tarfile.TarFile
    ):
        with io.BytesIO() as b:
            np.save(b, array)
            tarinfo = tarfile.TarInfo(file_name)
            tarinfo.size = len(b.getvalue())
            tarinfo.mtime = int(datetime.now().timestamp())
            b.seek(0)
            file.addfile(tarinfo, b)

    @abstractmethod
    def _load_calibration(self, file: tarfile.TarFile):
        raise NotImplementedError()

    def _load_image_file(self, file_path: str):
        return self._camera._load_image_file(file_path)

    def _load_video(self, file_path: str):
        return self._camera._load_video(file_path)

    def _next_video_frame(self):
        return self._camera._next_video_frame()

    def _preprocess_image(self, image: np.ndarray):
        return self._camera._preprocess_image(image)

    def _process_directory(self, file_path: str):
        return self._camera._process_directory(file_path)

    def _read_numpy_array(self, buffer: io.BufferedReader):
        with io.BytesIO(buffer.read()) as b:
            return np.load(b)

    def _seek_video_file(self, index: int):
        return self._seek_video_file(index)

    def _write_image_file(self, image: np.ndarray, file_path: str):
        self._write_image_file(image, file_path)

    @abstractmethod
    def _save_calibration(self, file: tarfile.TarFile):
        raise NotImplementedError()

    def _throw_error_if_none(self, value: any, error: str):
        if value is None:
            raise ValueError(error)

        return value

    def __iter__(self):
        return self

    def __next__(self):
        return self._camera.__next__()
