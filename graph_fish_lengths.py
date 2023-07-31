import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from array_read_write import read_camera_calibration, read_laser_calibration
from fish_mass import get_fish_lengths_and_masses
from constants import *

def prep_args() -> argparse.Namespace: 
    parser = argparse.ArgumentParser(prog='graph_fish_lengths',
                                     description='Graphing tool for graphing fish length measurements')
    parser.add_argument('-c', '--camcalib', help='Camera calibration file', dest='camera_calib_path', required=True)
    parser.add_argument('-l', '--lasercalib', help='Laser calibration file', dest='laser_calib_path', required=True)
    parser.add_argument('-i', '--input', help='Input csv files', dest='csv_file_list', nargs='+', required=True)
    parser.add_argument('-L', '--label', help='Labels for each dataset', dest='labels', nargs='+', required=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__": 
    args = prep_args()

    laser_position, laser_orientation = read_laser_calibration(args.laser_calib_path)
    camera_mat, _ = read_camera_calibration(args.camera_calib_path)
    focal_length_mm = camera_mat[0][0] * PIXEL_PITCH_MM
    sensor_size_px = np.array([4000,3000])
    principal_point = camera_mat[:2,2]
    camera_params = (focal_length_mm, sensor_size_px[0], sensor_size_px[1], PIXEL_PITCH_MM)

    all_lengths = []
    all_depths = []
    for csv in args.csv_file_list: 
        print(csv)
        depths, lengths, _ = get_fish_lengths_and_masses(file_name=csv, 
                                                               laser_position=laser_position, 
                                                               laser_orientation=laser_orientation,
                                                camera_params=camera_params,
                                                principal_point=principal_point) 

        all_lengths.append(lengths)
        all_depths.append(depths)

    reference = np.zeros(all_depths[0].shape)
    reference[:] = 0.31
    for i in range(len(all_lengths)):
        if args.labels:
            plt.plot(all_depths[i], all_lengths[i], '.', label=args.labels[i])
        else: 
            plt.plot(all_depths[i], all_lengths[i], '.')
    plt.plot(all_depths[0], reference, label='Reference')
    plt.xlabel('Distance from camera (m)')
    plt.ylabel('Measured fish length (m)')
    plt.title('Estimated fish lengths')
    plt.grid()
    plt.ylim(bottom=0, top=0.4)
    plt.legend()
    plt.show()