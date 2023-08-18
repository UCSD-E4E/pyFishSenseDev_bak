import os
import glob
import cv2
from PIL import Image
from enum import Enum
import argparse
from array_read_write import read_camera_calibration
from constants import *
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def prep_args():
    parser = argparse.ArgumentParser(prog='undistort_image',
                                     description='Rectifies recursive folder of pngs')
    parser.add_argument('-c', '--camcalib', help='Camera calibration file', dest='camera_calib_path', required=True)
    args = parser.parse_args()
    return args

def undistort_raw(tup):
    image_path, calibration_mat, distortion_coeffs = tup
    dst_path = os.path.join('./data/png_rectified','/'.join(image_path.split('/')[3:]))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return
    img = cv2.imread(image_path)
    try: 
        undistorted_img = cv2.undistort(img, calibration_mat, distortion_coeffs)
        cv2.imwrite(dst_path, undistorted_img)
    except: 
        return
        


if __name__ == "__main__":
    args = prep_args()
    calibration_mat, distortion_coeffs = read_camera_calibration(args.camera_calib_path)
    image_paths = glob.glob('./data/png/07/**/*.PNG', recursive=True)
    os.makedirs(os.fspath('./data/png_rectified'), exist_ok=True)
    with Pool(processes=cpu_count()) as pool: 
        list(tqdm(pool.imap(undistort_raw, [(file, calibration_mat, distortion_coeffs) for file in image_paths]), total=len(image_paths)))
        
