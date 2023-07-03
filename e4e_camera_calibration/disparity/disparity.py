from abc import ABC, abstractmethod
from typing import List
import math

from sherlock import Sherlock
import cv2
import numpy as np
import matplotlib.pyplot as plt

from e4e_camera_calibration.cameras.calibrated_stereo_camera import (
    CalibratedStereoCamera,
)


class DisparityBase(ABC):
    def __init__(self, calibrated_camera: CalibratedStereoCamera) -> None:
        super().__init__()

        self._calibrated_camera = calibrated_camera
        self._prev_min_error = math.inf

        plt.ion()

    def calibrate(
        self,
        calibration_directory: str,
        rows=14,
        columns=10,
        square_size=4.1 * 10**-3,
        display_calibration_error=False,
        max_error=0.5,
    ):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self._calibrated_camera.load(calibration_directory)

        left_images, right_images = zip(*self._calibrated_camera)

        left_images = [left_images[0]]
        right_images = [right_images[0]]

        line_segments_collection = []
        for left in left_images:
            points = self._get_points(left, rows, columns, criteria)
            line_segments_collection.append(
                list(self._get_line_segments(points, rows, columns))
            )

        X = self._get_inputs()
        count, _ = X.shape
        y = np.zeros((count, 2))
        budget = int(float(count) * 0.3)

        sherlock = Sherlock(
            n_init=5,
            budget=budget,
            surrogate_type="rbfthin_plate-rbf_multiquadric-randomforest-gpy",
            kernel="matern",
            num_restarts=0,
            pareto_margin=0,
            y_hint=None,
            plot_design_space=False,
            action_only=None,
            n_hint_init=0,
            scale_output=True,
            use_trace_as_prior=True,
            model_selection_type="mab10",
            request_output=lambda y, idx: self._step(
                X,
                y,
                idx,
                left_images,
                right_images,
                line_segments_collection,
                square_size,
                display_calibration_error,
            ),
        )
        sherlock.fit(X).predict(X, y)

    def _step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        known_idx: List[int],
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
        line_segments_collection: List[List[np.ndarray]],
        square_size: float,
        display_calibration_error: bool,
    ):
        for idx in known_idx:
            print(idx)

            errors = []

            array = X[idx, :]
            stereo = self._get_stereo(array)
            for left, right, line_segments in zip(
                left_images, right_images, line_segments_collection
            ):
                disparity = stereo.compute(left, right)
                depth = cv2.reprojectImageTo3D(
                    disparity, self._calibrated_camera.Q
                ).astype(np.float64)

                errors.extend(self._calculate_error(depth, line_segments, square_size))

            errors = np.array(errors)
            error = np.sqrt(np.sum(errors) / errors.shape[0])

            if display_calibration_error:
                print(f"Error: {error}, % Error: {error / square_size * 10:.2f}")

            if error < self._prev_min_error:
                self._prev_min_error = error

                np.save("disparity-parameters.npy", array)
                cv2.imwrite("disparity-image.png", array)

            if error == 0:
                y[idx, 0] = 0
            elif not math.isnan(error) and not math.isinf(error):
                y[idx, 0] = 1 / error
            else:
                y[idx, 0] = 0

    def _calculate_error(
        self, depth: np.ndarray, line_segments: List[np.ndarray], square_size: float
    ):
        errors = []

        for line_segment in line_segments:
            point1 = self._get_point(depth, line_segment[0])
            point2 = self._get_point(depth, line_segment[1])

            dist = np.linalg.norm(point1 - point2) ** 2
            errors.append(square_size - dist)

        return errors

    def _get_plane(self, points: List[np.ndarray]):
        x1, y1, z1 = points[0]
        x2, y2, z2 = points[1]
        x3, y3, z3 = points[2]

        a1 = x2 - x1
        b1 = y2 - y1
        c1 = z2 - z1
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = z3 - z1

        a = b1 * c2 - b2 * c1
        b = a2 * c1 - a1 * c2
        c = a1 * b2 - b1 * a2
        d = -a * x1 - b * y1 - c * z1

        return (a, b, c, d)

    def _get_point(self, depth: np.ndarray, point: np.ndarray):  # TOOD
        # floor_point = np.floor(point).astype(int)
        # ceil_point = np.ceil(point).astype(int)

        # corner1_2d = floor_point
        # corner2_2d = np.array([floor_point[1], ceil_point[0]])
        # corner3_2d = np.array([ceil_point[1], floor_point[0]])
        # corner4_2d = ceil_point

        # corner1_3d = depth[corner1_2d[1], corner1_2d[0], :]
        # corner2_3d = depth[corner2_2d[1], corner2_2d[0], :]
        # corner3_3d = depth[corner3_2d[1], corner3_2d[0], :]
        # corner4_3d = depth[corner4_2d[1], corner4_2d[0], :]

        # a, b, c, d = self._get_plane([corner1_3d, corner2_3d, corner3_3d])

        # delta_2d = point - floor_point
        # delta_3d = (corner1_3d - corner4_3d)[:2]

        # corner1_3d[:2] += delta_3d * delta_2d
        # corner1_3d[2] = -(a * corner1_3d[1] + b * corner1_3d[0] + d) / c

        # return corner1_3d

        return depth[int(point[1]), int(point[0]), :]

    def _get_points(self, image: np.ndarray, rows: int, columns: int, criteria):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret:

            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            # cv2.drawChessboardCorners(image, (rows, columns), corners, ret)
            # cv2.imshow("img", image)
            # k = cv2.waitKey(-1)

        return corners[:, 0, :]

    def _test(self, arr):
        if arr.shape == 1:
            raise NotImplementedError()

        return arr

    def _get_line_segments(self, points: List[np.ndarray], rows: int, columns: int):
        lines = []
        for c in range(columns):
            line_points = points[c * rows : (c + 1) * rows, :]
            lines.append(line_points)

        for line in lines:
            for i in range(0, rows, 2):
                yield self._test(line[i : i + 2])

            for i in range(1, rows, 2):
                yield self._test(line[i : i + 2])

        for i, line in enumerate(lines[:-1]):
            first_line = line
            second_line = lines[i + 1]

            for j, _ in enumerate(first_line):
                yield self._test(np.array([first_line[j], second_line[j]]))

    @abstractmethod
    def _get_inputs(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _get_stereo(self, idx: int, X: np.ndarray):
        raise NotImplementedError()


class SGBMDisparity(DisparityBase):
    def __init__(self, calibrated_camera: CalibratedStereoCamera) -> None:
        super().__init__(calibrated_camera)

    def _get_num_disparities(self):
        return np.arange(16, 192 + 16, 16)

    def _get_block_size(self):
        return np.arange(3, 11 + 2, 2)

    def _get_disp_12_max_diff(self):
        return np.arange(-1, 3)

    def _get_pre_filter_cap(self):
        return np.arange(0, 3)

    def _get_uniqueness_ratio(self):
        return np.arange(0, 3)

    def _get_speckle_window_size(self):
        return np.concatenate([np.array([0]), np.arange(50, 200 + 50, 50)])

    def _get_speckle_range(self):
        return np.arange(1, 3)

    def _get_inputs(self):
        input_options = [
            self._get_num_disparities(),
            self._get_block_size(),
            self._get_disp_12_max_diff(),
            self._get_pre_filter_cap(),
            self._get_uniqueness_ratio(),
            self._get_speckle_window_size(),
            self._get_speckle_range(),
        ]
        X = np.array(np.meshgrid(*input_options)).T.reshape(-1, len(input_options))

        return X

    def _get_stereo(self, array: np.ndarray):
        num_channels = 3
        num_disparities = array[0]
        block_size = array[1]
        disp_12_max_diff = array[2]
        pre_filter_cap = array[3]
        uniqueness_ratio = array[4]
        speckle_window_size = array[5]
        speckle_range = array[6]
        P1 = 8 * num_channels * block_size**2
        P2 = 32 * num_channels * block_size**2

        return cv2.StereoSGBM_create(
            numDisparities=num_disparities,
            blockSize=block_size,
            disp12MaxDiff=disp_12_max_diff,
            preFilterCap=pre_filter_cap,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            P1=P1,
            P2=P2,
        )
