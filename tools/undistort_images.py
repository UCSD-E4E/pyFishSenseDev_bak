import glob
import os
from enum import Enum

import cv2
from PIL import Image
from tqdm import tqdm

from util.constants import *
from util.array_read_write import read_camera_calibration


class ImageType(Enum):
    LASER_IMG = {
        'src': os.fspath("./data/fsl-01d-07-25-23/laser/"),
        'dst': os.fspath("./data/fsl-01d-07-25-23/laser-rectified/")
    }
    # FISH_IMG = {
    #     'src': os.fspath("/Users/kylehu/Desktop/4-19-2023 pool data/fish"),
    #     'dst': os.fspath("/Users/kylehu/Desktop/4-19-2023 pool data/fish-rectified"),
    # }

def raw_to_png(img_type: ImageType):
    # rawData = open(path, 'rb').read()
    # imgSize = (3000,4000) # the image size
    # img = Image.frombytes('RGB', imgSize, rawData)
    # img.save("foo.png")# can give any format you like .png
    pass
    
if __name__ == "__main__":
    calib_file = os.fspath('fsl-01d-lens-raw.dat')
    calibration_mat, distortion_coeffs = read_camera_calibration(calib_file)
    for image_type in (ImageType):
        src_path = image_type.value['src'] 
        dst_path = image_type.value['dst']
        filenames = glob.glob(os.path.join(src_path,'*.PNG'))
        for filename in tqdm(filenames):
            # print(f"Processing {filename}")
            img = cv2.imread(filename)
            undistorted_img = cv2.undistort(img, calibration_mat, distortion_coeffs)
            cv2.imwrite(os.path.join(dst_path, os.path.basename(filename)), undistorted_img)

