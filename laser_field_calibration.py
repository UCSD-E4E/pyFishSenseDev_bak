from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
from math import sin, cos
from array_read_write import read_camera_calibration
import glob
from constants import *

def error(img_contour: np.ndarray, slate_contour: np.ndarray):
    return scipy.spatial.distance.cdist(slate_contour.astype(float), img_contour.astype(float)).min(axis=1).mean() ** 2

def homo2contour(homogeneous_coords: np.ndarray):
    return np.round(homogeneous_coords[:, :2]).astype(int)

def calculate_transform(img_contour: np.ndarray, slate_contour: np.ndarray, angle: float):
    T = np.array([[cos(angle), -sin(angle), 0],
                  [sin(angle), cos(angle), 0],
                  [0, 0, 1]], dtype=float)
    
    slate_height, _, _ = slate_contour.shape
    slate_contour_homo = np.ones((slate_height, 3), dtype=float)
    slate_contour_homo[:, :2] = slate_contour.squeeze(1)

    transformed_slate_contour = (T @ slate_contour_homo.T).T

    img_radius = np.sqrt(cv2.contourArea(img_contour) / np.pi)
    slate_radius = np.sqrt(cv2.contourArea(homo2contour(transformed_slate_contour)) / np.pi)

    scale = img_radius / slate_radius

    T *= scale
    T[2, 2] = 1

    transformed_slate_contour = (T @ slate_contour_homo.T).T

    img_M = cv2.moments(img_contour)
    img_cx = int(img_M['m10']/img_M['m00'])
    img_cy = int(img_M['m01']/img_M['m00'])
    img_center = np.array([[img_cx], [img_cy]])

    slate_M = cv2.moments(homo2contour(transformed_slate_contour))
    slate_cx = int(slate_M['m10']/slate_M['m00'])
    slate_cy = int(slate_M['m01']/slate_M['m00'])
    slate_center = np.array([[slate_cx], [slate_cy]])

    T[:2, 2] = (img_center - slate_center).squeeze()

    return T

def get_slate_contour(slate_img: np.ndarray) -> np.ndarray:
    _, slate_threshold = cv2.threshold(slate_img, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5),np.uint8)
    opened = cv2.morphologyEx(slate_threshold, cv2.MORPH_OPEN, kernel)
    slate_contours, slate_hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    slate_contour = None
    for c in slate_contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            slate_contour = c
    return slate_contour

def get_slate_transformation(slate_img: np.ndarray, calib_img: np.ndarray) -> np.ndarray:
    '''Returns 2D transformation from reference slate to calibration image.

    Inputs; 
    slate_img: image of the reference slate. Assumed grayscale.
    calib_img: laser calibration image. Assumed grayscale.
    '''
    # get the largest contour from the original slate pdf
    slate_contour = get_slate_contour(slate_img)
    display = cv2.drawContours(np.zeros(slate_img.shape), [slate_contour], 0, 255, cv2.FILLED, 8)


    ret, thresholded = cv2.threshold(calib_img, 50, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(thresholded)
    plt.show()

    # get the matching contour in the calibration image
    test_contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_score = np.inf
    slate_in_image = None
    for t in test_contours: 
        score = cv2.matchShapes(t, slate_contour, cv2.CONTOURS_MATCH_I2, 0.0)
        if score < min_score: 
            min_score = score
            slate_in_image = t

    img_contour = cv2.approxPolyDP(slate_in_image, 20, closed=True)
    display = cv2.drawContours(np.zeros(calib_img.shape), [img_contour], 0, 255, cv2.FILLED, 8)
    plt.imshow(display)
    plt.show()

    slate_height, _, _ = slate_contour.shape
    slate_contour_homo = np.ones((slate_height, 3), dtype=float)
    slate_contour_homo[:, :2] = slate_contour.squeeze(1)

    _, _, img_angle = cv2.fitEllipse(img_contour)
    _, _, slate_angle = cv2.fitEllipse(slate_contour)
    angle_difference_one = (img_angle - slate_angle) * np.pi / 180.0

    T_one = calculate_transform(img_contour, slate_contour, angle_difference_one)
    T_two = calculate_transform(img_contour, slate_contour, np.pi + angle_difference_one)

    transformed_slate_contour_one = (T_one @ slate_contour_homo.T).T
    transformed_slate_contour_two = (T_two @ slate_contour_homo.T).T

    error_one = error(img_contour.squeeze(1), homo2contour(transformed_slate_contour_one))
    error_two = error(img_contour.squeeze(1), homo2contour(transformed_slate_contour_two))

    transformed_slate_contour = transformed_slate_contour_one if error_one < error_two else transformed_slate_contour_two
    T = T_one if error_one < error_two else T_two

    display = cv2.drawContours(np.zeros(calib_img.shape), [img_contour], 0, 255, cv2.FILLED, 8)
    pdf_display = cv2.drawContours(np.zeros(calib_img.shape), [homo2contour(transformed_slate_contour)], 0, 255, cv2.FILLED, 8)
    plt.imshow(np.abs(pdf_display - display))
    plt.show()

    return T

def get_dive_slate_pose(ref_img_path: str, img_folder_path: str, camera_calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
    '''Returns laser parameters for a given reference image and a list of images
    '''

    camera_mat, dist_coeffs = read_camera_calibration(camera_calib_path)

    glob_list = glob.glob(img_folder_path + '*.PNG')
    glob_list.sort()
    test_img = glob_list[2]
    print(test_img)
    test_img = cv2.imread(Path(test_img).absolute().as_posix(), cv2.IMREAD_GRAYSCALE)


    slate_img = cv2.imread(Path(ref_img_path).absolute().as_posix(), cv2.IMREAD_GRAYSCALE)
    T = get_slate_transformation(slate_img, test_img)

    # get points from the reference slate
    slate_points = get_slate_contour(slate_img)
    slate_height, _, _ = slate_points.shape
    slate_contour_homo = np.zeros((slate_height, 3), dtype=float)
    slate_contour_homo[:, :2] = slate_points.squeeze(1)
    slate_contour_obj = slate_contour_homo
    slate_contour_obj[:2] = 0

    transformed_points_homo = T @ slate_contour_homo.T
    transformed_points = transformed_points_homo[...,:2]/transformed_points_homo[...,2]

    empty_dist_coeffs=  np.zeros((5,))
    ret, rvecs, tvecs = cv2.solvePnP(slate_contour, transformed_points, camera_mat, empty_dist_coeffs)

    return rvecs, tvecs

if __name__ == "__main__":
    slate_path = "./data-8-3-florida/png_rectified/slate/slate.jpg"
    image_folder_path = "./data-8-3-florida/png_rectified/H slate dive 3/"
    camera_calib_path = ""
    r, t = get_dive_slate_pose(slate_path, image_folder_path, camera_calib_path)
