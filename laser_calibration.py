import cv2
import numpy as np
import csv
from laser_parallax import compute_world_points_from_depths
from array_read_write import read_camera_calibration, write_laser_calibration
import os
from constants import *

def get_jacobian(
        ps: np.ndarray, 
        state: np.ndarray) -> np.ndarray:
    n = ps.shape[0]
    jacobian = np.zeros((n * 3, 5))
    alpha = state[:3]
    l = np.array([state[3], state[4], 0])
    for i in range(n):
        p = ps[i]
        jacobian[3*i:3*i+3,:3] = np.eye(3) * np.linalg.norm(p - l)
        jacobian[3*i:3*i+3,3:5] = (alpha[:,np.newaxis] @ (p - l)[np.newaxis,:2]) / np.linalg.norm(p - l)
        jacobian[[3*i, 3*i+1],[3,4]] *= -1
        jacobian[[3*i,3*i+1],[3,4]] += 1
    return jacobian

def get_residual(
    points: np.ndarray,
    state: np.ndarray) -> float:
    laser_pos = np.array([state[3], state[4], 0])
    laser_angle = state[:3]
    est_points = np.linalg.norm(points - laser_pos, axis=1)[:, np.newaxis] * laser_angle + laser_pos
    return points - est_points

def gauss_newton_estimate_state(
        ps: np.ndarray, 
        init_state: np.ndarray, 
        num_iterations: int =10) -> np.ndarray:
    state = init_state
    residual_norm_squared = []
    for _ in range(num_iterations):
        J = get_jacobian(ps, state)
        rs = get_residual(ps, state).flatten()
        residual_norm_squared.append(np.dot(rs,rs)) 
        temp_state = state + np.linalg.pinv(J) @ rs
        state = temp_state
    return state, residual_norm_squared

if __name__ == "__main__":
    # read camera calibration parameters from file
    camera_mat, dist_coeffs  = read_camera_calibration('calibration-output.dat')
    focal_length_mm = camera_mat[0][0] * PIXEL_PITCH_MM
    sensor_size_px = np.array([4000,3000])
    camera_params = (focal_length_mm, sensor_size_px[0], sensor_size_px[1], PIXEL_PITCH_MM)

    # load in undistorted laser images
    img_coords = []
    file_list = []
    with open(os.fspath('./laser_data.csv'),'r',encoding='utf-8-sig') as csvfile: 
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader: 
            file_list.append(row['name'])
            img_coords.append([int(row['x']),int(row['y'])])

    img_coords = np.array(img_coords)

    # determine calibration board pose for each photo
    # find laser dot 3D coordinates for each photo
    depths = []
    objp = np.zeros((14*10,3), np.float32)
    objp[:,:2] = np.mgrid[0:14,0:10].T.reshape(-1,2)
    for i,file_name in enumerate(file_list): 
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find checkerboard
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, corners = cv2.findChessboardCorners(img, (14,10), None)
        if not ret:
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        corners2 = np.squeeze(corners2)
        # Find the rotation and translation vectors from object frame to camera frame
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_mat, dist_coeffs)

        # print(rvecs, tvecs)
        
        # find the plane that the laser passes through by converting checkerboard points to camera frame using the rotation and translation vectors
        rmat, _ = cv2.Rodrigues(rvecs)
        checkerboard_to_meter = CHECKERBOARD_SQUARE_SIZE_MM * MM_TO_M
        board_plane_points = ((rmat @ objp[[0,1,14]].T + tvecs) * checkerboard_to_meter).T
        print(f"Estimated depth of checkerboard in {os.path.basename(file_name)} is {board_plane_points[0,2]}")
        normal_vec = np.cross(board_plane_points[1] - board_plane_points[0], board_plane_points[2] - board_plane_points[0])

        # define laser ray assuming pinhole camera
        laser_ray = np.zeros(3)
        laser_ray[:2] = (sensor_size_px/2 - img_coords[i]) * PIXEL_PITCH_MM * MM_TO_M
        laser_ray[2] = -focal_length_mm * MM_TO_M

        # find scale factor such that the laser ray intersects with the plane
        scale_factor = (normal_vec.T @ board_plane_points[0])/(normal_vec.T @ laser_ray) 
        laser_3d_cam = laser_ray * scale_factor

        depths.append(laser_3d_cam[2])

    depths = np.array(depths)
    # use list of laser dot coordinates to calibrate laser
    ps = compute_world_points_from_depths(
        camera_params=camera_params,
        image_coordinates=(sensor_size_px/2 - img_coords),
        depths=depths
    )

    state_init = np.array([0,0,1,-0.04,-0.11])
    state, _ = gauss_newton_estimate_state(ps, state_init)

    # save the states to a file
    laser_axis = state[:3]
    laser_pos = np.zeros((3,))
    laser_pos[:2] = state[3:5]

    file_path = "laser-calibration-output.dat"
    write_laser_calibration(file_path, laser_axis, laser_pos)
    