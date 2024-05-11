from math import cos, sin
from pathlib import Path
from typing import Any, Tuple

import cv2
import fitz
import matplotlib.pyplot as plt
import numpy as np
import scipy

from pyfishsensedev.laser.calibration.plane_detector import PlaneDetector
from pyfishsensedev.library.array_read_write import read_camera_calibration


class SlatePlaneDetector(PlaneDetector):
    def __init__(self, slate_pdf_path: Path, lens_calibration_path: Path) -> None:
        super().__init__()

        self.calibration_matrix, self.distortion_coeffs = read_camera_calibration(
            lens_calibration_path.as_posix()
        )

        doc = fitz.open(slate_pdf_path.as_posix())
        page = doc.load_page(0)
        self.pix = page.get_pixmap()
        self.slate_img = cv2.cvtColor(
            np.frombuffer(buffer=self.pix.samples, dtype=np.uint8).reshape(
                (self.pix.height, self.pix.width, 3)
            ),
            cv2.COLOR_BGR2GRAY,
        )

        # get the largest contour from the original slate pdf
        ret2, slate_threshold = cv2.threshold(
            self.slate_img, 100, 255, cv2.THRESH_BINARY_INV
        )
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(slate_threshold, cv2.MORPH_OPEN, kernel)
        slate_contours, slate_hierarchy = cv2.findContours(
            opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        max_area = 0
        self.slate_contour: Any = None
        for c in slate_contours:
            area = cv2.contourArea(c)
            if area > max_area:
                max_area = area
                self.slate_contour = c

    def _error(self, img_contour: np.ndarray, slate_contour: np.ndarray):
        return (
            scipy.spatial.distance.cdist(
                slate_contour.astype(float), img_contour.astype(float)
            )
            .min(axis=1)
            .mean()
            ** 2
        )

    def _homo2contour(self, homogeneous_coords: np.ndarray):
        return np.round(homogeneous_coords[:, :2]).astype(int)

    def _calculate_transform(
        self, img_contour: np.ndarray, slate_contour: np.ndarray, angle: float
    ):
        T = np.array(
            [[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]],
            dtype=float,
        )

        slate_height, _, _ = slate_contour.shape
        slate_contour_homo = np.ones((slate_height, 3), dtype=float)
        slate_contour_homo[:, :2] = slate_contour.squeeze(1)

        transformed_slate_contour = (T @ slate_contour_homo.T).T

        img_radius = np.sqrt(cv2.contourArea(img_contour) / np.pi)
        slate_radius = np.sqrt(
            cv2.contourArea(self._homo2contour(transformed_slate_contour)) / np.pi
        )

        scale = img_radius / slate_radius

        T *= scale
        T[2, 2] = 1

        transformed_slate_contour = (T @ slate_contour_homo.T).T

        img_M = cv2.moments(img_contour)
        img_cx = int(img_M["m10"] / img_M["m00"])
        img_cy = int(img_M["m01"] / img_M["m00"])
        img_center = np.array([[img_cx], [img_cy]])

        slate_M = cv2.moments(self._homo2contour(transformed_slate_contour))
        slate_cx = int(slate_M["m10"] / slate_M["m00"])
        slate_cy = int(slate_M["m01"] / slate_M["m00"])
        slate_center = np.array([[slate_cx], [slate_cy]])

        T[:2, 2] = (img_center - slate_center).squeeze()

        return T

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        plt.imshow(thresholded)
        plt.show()

        slate_height, _, _ = self.slate_contour.shape
        slate_contour_homo = np.ones((slate_height, 3), dtype=float)
        slate_contour_homo[:, :2] = self.slate_contour.squeeze(1)

        # get the matching contour in the calibration image
        test_contours, hierarchy = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_score = np.inf
        slate_in_image = None
        for t in test_contours:
            score = cv2.matchShapes(t, self.slate_contour, cv2.CONTOURS_MATCH_I2, 0.0)
            if score < min_score:
                min_score = score
                slate_in_image = t

        img_contour: np.ndarray = cv2.approxPolyDP(slate_in_image, 10, closed=True)

        _, _, img_angle = cv2.fitEllipse(img_contour)
        _, _, slate_angle = cv2.fitEllipse(self.slate_contour)

        angle_difference_one = (img_angle - slate_angle) * np.pi / 180.0

        T_one = self._calculate_transform(
            img_contour, self.slate_contour, angle_difference_one
        )
        T_two = self._calculate_transform(
            img_contour, self.slate_contour, np.pi + angle_difference_one
        )

        transformed_slate_contour_one = (T_one @ slate_contour_homo.T).T
        transformed_slate_contour_two = (T_two @ slate_contour_homo.T).T

        error_one = self._error(
            img_contour.squeeze(1), self._homo2contour(transformed_slate_contour_one)
        )
        error_two = self._error(
            img_contour.squeeze(1), self._homo2contour(transformed_slate_contour_two)
        )

        transformed_slate_contour = (
            transformed_slate_contour_one
            if error_one < error_two
            else transformed_slate_contour_two
        )
        T = T_one if error_one < error_two else T_two

        img_corners = scipy.spatial.distance.cdist(
            img_contour.squeeze(1).astype(float),
            self._homo2contour(transformed_slate_contour).astype(float),
            "euclidean",
        ).argmin(axis=1)

        slate_coords = np.zeros((img_corners.shape[0], 3))
        slate_coords[:, :2] = self.slate_contour[img_corners, 0, :]

        empty_dist_coeffs = np.zeros((5,))
        ret, rvecs, tvecs = cv2.solvePnP(
            self.slate_coords,
            img_contour.squeeze(1).astype(float),
            self.calibration_matrix,
            empty_dist_coeffs,
        )
        rmat, _ = cv2.Rodrigues(rvecs)

        pixel_size = 0.0254 / self.pix.xres
        board_plane_points = ((rmat @ slate_coords.T + tvecs) * pixel_size).T
        normal_vec = np.cross(
            board_plane_points[1] - board_plane_points[0],
            board_plane_points[2] - board_plane_points[0],
        )

        return board_plane_points, normal_vec
