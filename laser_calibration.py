import cv2
import numpy as np
import csv
from array_read_write import read_camera_calibration, write_laser_calibration
import os
from constants import *
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

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

def atanasov_method(
        ps: np.ndarray
):
    '''
    Nikolay's method for laser calibration.
    Inputs:
     - ps: the laser points
    Output: the 5-vector of the laser parameters, with the first 3 being the orientation,
            and the final two being the x and y coordinates of the laser origin
    '''
    avg_alpha = np.zeros((3,))
    params = np.zeros((5,))
    for i in range(ps.shape[0]):
        for j in range(ps.shape[0]):
            if i != j:
                v = ps[i] - ps[j]
                if v[2] < 0: v = -v
                avg_alpha += v

    avg_alpha /= np.linalg.norm(avg_alpha)
    if avg_alpha[2] < 0:
        avg_alpha = -avg_alpha

    centroid = np.mean(ps,axis=0)
    scale_factor = centroid[2]/avg_alpha[2]
    params[0:3] = avg_alpha
    params[3:5] = centroid[0:2] - scale_factor * avg_alpha[0:2]
    return params


def prep_args() -> argparse.ArgumentParser: 
    parser = argparse.ArgumentParser(prog='laser_calibration',
                                     description='Given a CSV of images file names and laser dot coordinates, generate laser calibration parameters'
                                     )
    parser.add_argument('-c', '--calib', help='Camera calibration file', dest='camera_calib_path', required=True)
    parser.add_argument('-i', '--input', help='Input csv file', dest='csv_path', required=True)
    parser.add_argument('-o', '--output', help='Output destination and file name', dest='dest_path', required=True)
    parser.add_argument('--usegn', help="Use Gauss-Newton optimization, otherwise will default to naive approach", action=argparse.BooleanOptionalAction, dest='use_gn')
    return parser

# return the depths
def find_laser_depth(tup):
    idx, file_name, img_coord, camera_mat, focal_length_mm, principal_point = tup
    print(f"Processing {os.path.basename(file_name)}")
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findChessboardCorners(img, (14,10), None)
    if not ret:
        return None, None
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    corners2 = np.squeeze(corners2)
    # Find the rotation and translation vectors from object frame to camera frame
    # distortion coefficients passed into solvePnP should be 0 since we are dealing with a undistorted image
    objp = np.zeros((14*10,3), np.float32)
    objp[:,:2] = np.mgrid[0:14,0:10].T.reshape(-1,2)
    empty_dist_coeffs = np.zeros((5,))
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_mat, empty_dist_coeffs)

    # find the plane that the laser passes through by converting checkerboard points to camera frame using the rotation and translation vectors
    rmat, _ = cv2.Rodrigues(rvecs)
    checkerboard_to_meter = CHECKERBOARD_SQUARE_SIZE_MM * MM_TO_M
    board_plane_points = ((rmat @ objp[[0,1,14]].T + tvecs) * checkerboard_to_meter).T
    normal_vec = np.cross(board_plane_points[1] - board_plane_points[0], board_plane_points[2] - board_plane_points[0])

    # define laser ray assuming pinhole camera
    laser_ray = np.zeros(3)
    laser_ray[:2] = (principal_point - img_coord) * PIXEL_PITCH_MM * MM_TO_M
    laser_ray[2] = -focal_length_mm * MM_TO_M

    # find scale factor such that the laser ray intersects with the plane
    scale_factor = (normal_vec.T @ board_plane_points[0])/(normal_vec.T @ laser_ray) 
    laser_3d_cam = laser_ray * scale_factor

    print(f"Estimated depth of laser dot in {os.path.basename(file_name)} is {laser_3d_cam[2]}")
    # depths.append(laser_3d_cam[2])
    # img_coords2.append(img_coords[i])                
    return idx, laser_3d_cam

if __name__ == "__main__":
    parser = prep_args()
    args = parser.parse_args()

    # read camera calibration parameters from file
    camera_mat, dist_coeffs  = read_camera_calibration(args.camera_calib_path)
    focal_length_mm = camera_mat[0][0] * PIXEL_PITCH_MM
    sensor_size_px = np.array([4000,3000])
    principal_point = camera_mat[:2,2]
    camera_params = (focal_length_mm, sensor_size_px[0], sensor_size_px[1], PIXEL_PITCH_MM)

    # load in undistorted laser images
    img_coords = []
    file_list = []
    with open(args.csv_path,'r',encoding='utf-8-sig') as csvfile: 
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader: 
            file_list.append(row['name'])
            img_coords.append([float(row['laser.x']),float(row['laser.y'])])

    # determine calibration board pose for each photo
    # find laser dot 3D coordinates for each photo
    depths = []
    img_coords2 = []
    # distance_list = []
    idxs = list(enumerate(file_list))[0]
    pool_args = [(i, file_list[i], img_coords[i], camera_mat, focal_length_mm, principal_point) for i in range(len(file_list))]
    with Pool(processes=4) as pool:
        result = list(tqdm(pool.imap(find_laser_depth, pool_args), total=len(file_list)))
    result = [(idx, point) for idx, point in result if idx is not None and point is not None]

    combined_result = list(zip(*result))
    _, ps = combined_result
    ps = np.array(ps)
    
    # use list of laser dot coordinates to calibrate laser
    if args.use_gn:
        state_init = np.array([0,0,1,-0.04,-0.11])
        state, _ = gauss_newton_estimate_state(ps, state_init)
    else:
        state = atanasov_method(ps)

    # save the states to a file
    laser_axis = state[:3]
    laser_pos = np.zeros((3,))
    laser_pos[:2] = state[3:5]

    print(f"Resulting axis and position: {laser_axis}, {laser_pos}")

    write_laser_calibration(args.dest_path, laser_axis, laser_pos)
