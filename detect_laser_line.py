import numpy as np
import cv2
from pathlib import Path
from array_read_write import read_laser_calibration, read_camera_calibration
import os

def get_2d_from_3d(calibration_matrix, vector):

    homogeneous_coords = vector/vector[2]
    return calibration_matrix @ homogeneous_coords

def get_line(one, two):

    slope =(two[1]-one[1]) / (two[0] - one[0])

    x1 = 0
    y1 = int(slope * (-one[0]) + one[1])

    if (y1 < 0):
        y1 = 0
        x1 = int((-one[1])/slope + one[0])

    # We can now calculate the point of convergence. 

    x2, y2 = two

    return (x1,y1), (x2,y2)

def return_line(laser_path: Path, calibration_path: Path):

    laser_position, laser_orientation = read_laser_calibration(laser_path.as_posix())
    calibration_matrix, _ = read_camera_calibration(calibration_path.as_posix())

    first_point = laser_position + 1 * laser_orientation
    second_point = laser_position + 10000 * laser_orientation

    first_2d = get_2d_from_3d(calibration_matrix, first_point)
    second_2d = get_2d_from_3d(calibration_matrix, second_point)


    first_2d_tup = (int(first_2d[0]), int(first_2d[1]))
    second_2d_tup = (int(second_2d[0]), int(second_2d[1]))

    return get_line(first_2d_tup, second_2d_tup)

def view_line(laser_path: Path, calibration_path: Path, data_path: Path):

    for file in os.listdir(data_path.as_posix()):
        
        filepath = data_path.joinpath(file)
        if filepath.suffix != ".JPG":
            continue
        
        img = cv2.imread(filepath.as_posix())

        start_point, end_point = return_line(laser_path, calibration_path)

        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)

        cv2.line(img, start_point, end_point, color=(255,255,255), thickness=40) 

        cv2.imshow("Resized_Window", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()