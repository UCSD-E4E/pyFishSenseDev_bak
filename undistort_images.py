import os
import glob
import cv2
from PIL import Image
from enum import Enum
from array_read_write import read_camera_calibration
from constants import *

class ImageType(Enum):
    LASER_IMG = {
        'src': os.fspath("/Users/kylehu/Desktop/4-19-2023 pool data/calibration"),
        'dst': os.fspath("/Users/kylehu/Desktop/4-19-2023 pool data/calibration-rectified")
    }
    FISH_IMG = {
        'src': os.fspath("/Users/kylehu/Desktop/4-19-2023 pool data/fish"),
        'dst': os.fspath("/Users/kylehu/Desktop/4-19-2023 pool data/fish-rectified"),
    }

def raw_to_png(img_type: ImageType):
    # rawData = open(path, 'rb').read()
    # imgSize = (3000,4000) # the image size
    # img = Image.frombytes('RGB', imgSize, rawData)
    # img.save("foo.png")# can give any format you like .png
    pass
    
if __name__ == "__main__":
    calib_file = os.fspath('calibration-output.dat')
    calibration_mat, distortion_coeffs = read_camera_calibration(calib_file)
    for image_type in (ImageType):
        src_path = image_type.value['src'] 
        dst_path = image_type.value['dst']
        filenames = glob.glob(os.path.join(src_path,'*.JPG'))
        for filename in filenames:
            img = cv2.imread(filename)
            undistorted_img = cv2.undistort(img, calibration_mat, distortion_coeffs)
            cv2.imwrite(os.path.join(dst_path, os.path.basename(filename)), undistorted_img)

