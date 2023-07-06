import cv2
import numpy as np
import csv
from laser_parallax import compute_world_points_from_depths
from array_read_write import read_camera_calibration
import io
import datetime
import tarfile
import json

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

def save_numpy_array(
        array: np.ndarray, file_name: str, file: tarfile.TarFile
    ):
    with io.BytesIO() as b:
        np.save(b, array)
        tarinfo = tarfile.TarInfo(file_name)
        tarinfo.size = len(b.getvalue())
        tarinfo.mtime = int(datetime.now().timestamp())
        b.seek(0)
        file.addfile(tarinfo, b)


if __name__ == "__main__":
    # read camera calibration parameters from file
    camera_mat, dist_coeffs  = read_camera_calibration('camera_calibration.tar')
    pixel_pitch_mm = 1.5*1e-3
    focal_length_mm = camera_mat[0][0] * pixel_pitch_mm
    sensor_size_px = np.array([4000,3000])
    camera_params = (focal_length_mm, sensor_size_px[0], sensor_size_px[1], pixel_pitch_mm)

    # load in undistorted laser images
    img_coords = []
    file_list = []
    with open('5-8-23/data.csv','r',encoding='utf-8-sig') as csvfile: 
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader: 
            file_list.append(row['name'])
            img_coords.append([int(row['x']),int(row['y'])])

    # determine calibration board pose for each photo
    # find laser dot 3D coordinates for each photo
    depths = []
    for i,file_name in enumerate(file_list): 
        img = cv2.imread(file_name)
        # find checkerboard
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, corners = cv2.findChessboardCorners(img, (14,10), None)
        objp = np.zeros((14*10,3), np.float32)
        objp[:,:2] = np.mgrid[0:14,0:10].T.reshape(-1,2)
        if not ret:
            continue

        corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
        # cv2.drawChessboardCorners(test_image, (14,10), corners2, ret)

        # CALIBRATION PARAMETERS REQUIRED BEYOND HERE
        # Find the rotation and translation vectors from object frame to camera frame
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_mat, dist_coeffs) 

        # convert laser dot image dot coordinates to checkerboard object frame
        checkerboard_x_vec = corners2[1] - corners2[0]
        angle = np.arctan2(checkerboard_x_vec[1], checkerboard_x_vec[0])
        rot = np.array([[np.cos(-angle), np.sin(-angle)],
                        [-np.sin(-angle), np.cos(-angle)]])
        square_length_px = np.linalg.norm(checkerboard_x_vec)
       
        laser_2d_obj = (rot @ (img_coords[i] - corners2[0])) / square_length_px
        laser_3d_obj = np.zeros(3)
        laser_3d_obj[:2] = laser_2d_obj

        # convert laser dot in checkerboard object frame to camera frame
        laser_3d_cam = cv2.projectPoints(laser_3d_obj, rvecs, tvecs, camera_mat, dist_coeffs)
        depths.append(laser_3d_cam[2])

    # use list of laser dot coordinates to calibrate laser
    ps = compute_world_points_from_depths(
        camera_params=camera_params,
        image_coordinates=(sensor_size_px/2 - img_coords),
        depths=depths/100
    )

    state_init = np.array([0,0,1,-0.04,-0.11])
    state, _ = gauss_newton_estimate_state(ps, state_init)

    # save the states to a file
    laser_axis = state[:3]
    laser_pos = np.zeros((3,))
    laser_pos[:2] = state[3:5]

    file_path = "laser_calibration_data.tar"
    metadata = {
        "calibrated_date": str(datetime.now()),
    }
    with tarfile.open(file_path, "x:gz") as f:
        with io.StringIO() as s:
            json.dump(metadata, s, indent=True)
            bytes = s.getvalue().encode("utf8")
            with io.BytesIO(bytes) as b:
                tarinfo = tarfile.TarInfo("metadata.json")
                tarinfo.size = len(bytes)
                tarinfo.mtime = int(datetime.now().timestamp())
                f.addfile(tarinfo, b)

        save_numpy_array(laser_axis, file_path, f)
        save_numpy_array(laser_pos, file_path, f) 