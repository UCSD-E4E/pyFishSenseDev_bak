from typing import List
import io
import tarfile

import cv2
import numpy as np

from e4e_camera_calibration.cameras.calibrated_camera_base import CalibratedCameraBase
from e4e_camera_calibration.cameras.stereo_camera import StereoCamera


class CalibratedStereoCamera(CalibratedCameraBase, StereoCamera):
    def __init__(self, camera: StereoCamera, **kwargs) -> None:
        super().__init__(camera, **kwargs)

        self._left_camera_matrix: np.ndarray = None
        self._right_camera_matrix: np.ndarray = None
        self._left_distortion_coefficients: np.ndarray = None
        self._right_distortion_coefficients: np.ndarray = None
        self._stereo_R: np.ndarray = None
        self._stereo_T: np.ndarray = None
        self._left_stereo_map_x: np.ndarray = None
        self._left_stereo_map_y: np.ndarray = None
        self._right_stereo_map_x: np.ndarray = None
        self._right_stereo_map_y: np.ndarray = None
        self._Q: np.ndarray = None

    @property
    def left_camera_matrix(self) -> np.ndarray:
        return self._throw_error_if_none(
            self._left_camera_matrix, "Left camera matrix is not initialized."
        )

    @property
    def left_distortion_coefficients(self) -> np.ndarray:
        return self._throw_error_if_none(
            self._left_distortion_coefficients,
            "Left distortion coefficients is not initialized.",
        )

    @property
    def right_camera_matrix(self) -> np.ndarray:
        return self._throw_error_if_none(
            self._right_camera_matrix, "Right camera matrix is not initialized."
        )

    @property
    def right_distortion_coefficients(self) -> np.ndarray:
        return self._throw_error_if_none(
            self._right_distortion_coefficients,
            "Right distortion coefficients is not initialized.",
        )

    @property
    def stereo_R(self):
        return self._stereo_R

    @property
    def stereo_T(self):
        return self._stereo_T

    @property
    def left_stereo_map_x(self):
        return self._left_stereo_map_x

    @property
    def left_stereo_map_y(self):
        return self._left_stereo_map_y

    @property
    def right_stereo_map_x(self):
        return self._right_stereo_map_x

    @property
    def right_stereo_map_y(self):
        return self._right_stereo_map_y

    @property
    def Q(self):
        return self._Q

    def calibrate(
        self,
        rows=14,
        columns=10,
        square_size=4.1 * 10**-3,
        display_calibration_error=False,
        max_error=0.5,
    ):
        left, right = list(zip(*self._camera))
        width = left[0].shape[1]
        height = left[0].shape[0]

        (
            left_error,
            self._left_camera_matrix,
            self._left_distortion_coefficients,
        ) = self._calibrate_camera(
            left, rows, columns, square_size, max_error=max_error, camera_name="Left"
        )

        (
            right_error,
            self._right_camera_matrix,
            self._right_distortion_coefficients,
        ) = self._calibrate_camera(
            right, rows, columns, square_size, max_error=max_error, camera_name="Right"
        )

        if display_calibration_error:
            print(
                f"Left Calibration Error: {left_error}\nRight Calibration Error: {right_error}"
            )

        error, self._stereo_R, self._stereo_T = self._stereo_calibration(
            left, right, rows, columns, square_size, max_error=max_error
        )

        if display_calibration_error:
            print(f"Stereo Calibration Error: {error}")

        (
            self._left_stereo_map_x,
            self._left_stereo_map_y,
            self._right_stereo_map_x,
            self._right_stereo_map_y,
            self._Q,
        ) = self._rectify_stereo_cameras(width, height)

    def _load_calibration(self, file: tarfile.TarFile):
        self._left_camera_matrix = self._read_numpy_array(
            file.extractfile("left_calibration_matrix.npy")
        )
        self._right_camera_matrix = self._read_numpy_array(
            file.extractfile("right_calibration_matrix.npy")
        )
        self._left_distortion_coefficients = self._read_numpy_array(
            file.extractfile("left_distortion_coefficients.npy")
        )
        self._right_distortion_coefficients = self._read_numpy_array(
            file.extractfile("right_distortion_coefficients.npy")
        )
        self._stereo_R = self._read_numpy_array(file.extractfile("stereo_R.npy"))
        self._stereo_T = self._read_numpy_array(file.extractfile("stereo_T.npy"))
        self._left_stereo_map_x = self._read_numpy_array(
            file.extractfile("left_stereo_map_x.npy")
        )
        self._left_stereo_map_y = self._read_numpy_array(
            file.extractfile("left_stereo_map_y.npy")
        )
        self._right_stereo_map_x = self._read_numpy_array(
            file.extractfile("right_stereo_map_x.npy")
        )
        self._right_stereo_map_y = self._read_numpy_array(
            file.extractfile("right_stereo_map_y.npy")
        )
        self._Q = self._read_numpy_array(file.extractfile("Q.npy"))

    def _save_calibration(self, file: tarfile.TarFile):
        self._add_numpy_array(
            self.left_camera_matrix, "left_calibration_matrix.npy", file
        )
        self._add_numpy_array(
            self.right_camera_matrix, "right_calibration_matrix.npy", file
        )
        self._add_numpy_array(
            self.left_distortion_coefficients, "left_distortion_coefficients.npy", file
        )
        self._add_numpy_array(
            self.right_distortion_coefficients,
            "right_distortion_coefficients.npy",
            file,
        )
        self._add_numpy_array(self.stereo_R, "stereo_R.npy", file)
        self._add_numpy_array(self.stereo_T, "stereo_T.npy", file)
        self._add_numpy_array(self.left_stereo_map_x, "left_stereo_map_x.npy", file)
        self._add_numpy_array(self.left_stereo_map_y, "left_stereo_map_y.npy", file)
        self._add_numpy_array(self.right_stereo_map_x, "right_stereo_map_x.npy", file)
        self._add_numpy_array(self.right_stereo_map_y, "right_stereo_map_y.npy", file)
        self._add_numpy_array(self.Q, "Q.npy", file)

    def _rectify_stereo_cameras(self, width: int, height: int):
        (
            left_rotation_matrix,
            right_rotation_matix,
            left_projection_matrix,
            right_projection_matrix,
            Q,
            roiL,
            roiR,
        ) = cv2.stereoRectify(
            self.left_camera_matrix,
            self.left_distortion_coefficients,
            self.right_camera_matrix,
            self.right_distortion_coefficients,
            (width, height),
            self.stereo_R,
            self.stereo_T,
            1,
            (0, 0),
        )

        left_stereo_map = cv2.initUndistortRectifyMap(
            self.left_camera_matrix,
            self.left_distortion_coefficients,
            left_rotation_matrix,
            left_projection_matrix,
            (width, height),
            cv2.CV_16SC2,
        )

        right_stereo_map = cv2.initUndistortRectifyMap(
            self.right_camera_matrix,
            self.right_distortion_coefficients,
            right_rotation_matix,
            right_projection_matrix,
            (width, height),
            cv2.CV_16SC2,
        )

        return (
            left_stereo_map[0],
            left_stereo_map[1],
            right_stereo_map[0],
            right_stereo_map[1],
            Q,
        )

    def _stereo_calibration(
        self,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
        rows: int,
        columns: int,
        square_size: float,
        max_error=0.5,
    ):
        # The following is adapted from https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        # coordinates of squares in the checkerboard world space
        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp = square_size * objp

        # frame dimensions. Frames should be the same size.
        width = left_images[0].shape[1]
        height = left_images[0].shape[0]

        # Pixel coordinates of checkerboards
        imgpoints_left = []  # 2d points in image plane.
        imgpoints_right = []

        # coordinates of the checkerboard in checkerboard world space.
        objpoints = []  # 3d point in real world space

        for left_img, right_img in zip(left_images, right_images):
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
            left_c_ret, left_corners = cv2.findChessboardCorners(
                left_gray, (rows, columns), None
            )
            right_c_ret, right_corners = cv2.findChessboardCorners(
                right_gray, (rows, columns), None
            )

            if left_c_ret and right_c_ret:
                left_corners = cv2.cornerSubPix(
                    left_gray, left_corners, (11, 11), (-1, -1), criteria
                )
                right_corners = cv2.cornerSubPix(
                    right_gray, right_corners, (11, 11), (-1, -1), criteria
                )

                objpoints.append(objp)
                imgpoints_left.append(left_corners)
                imgpoints_right.append(right_corners)

        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC

        (
            error,
            left_camera_matrix,
            left_distortion_coefficients,
            right_camera_matrix,
            right_distortion_coefficients,
            R,
            T,
            E,
            F,
        ) = cv2.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            self.left_camera_matrix,
            self.left_distortion_coefficients,
            self.right_camera_matrix,
            self.right_distortion_coefficients,
            (width, height),
            criteria=criteria,
            flags=stereocalibration_flags,
        )

        if error > max_error:
            raise ValueError(
                f"An error of {error} was above the maximum allowed error of {max_error} while performing stereo calibration."
            )

        return error, R, T

    def __next__(self):
        left, right = super().__next__()
        left_rect = np.zeros_like(left)
        right_rect = np.zeros_like(right)

        for i in range(3):
            left_rect[:, :, i] = cv2.remap(
                left[:, :, i],
                self.left_stereo_map_x,
                self.left_stereo_map_y,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0,
            )

            right_rect[:, :, i] = cv2.remap(
                right[:, :, i],
                self.right_stereo_map_x,
                self.right_stereo_map_y,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
                0,
            )

        left_rect = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)
        right_rect = cv2.cvtColor(right_rect, cv2.COLOR_BGR2RGB)

        return left_rect, right_rect
