import os
import glob
import cv2
from PIL import Image
from enum import Enum
from array_read_write import read_camera_calibration

class ImageType(Enum):
    LASER_IMG = {
        'src': os.fspath("./data/laser_jpgs"),
        'dst': os.fspath("./data/laser_jpgs_rectified")
    }
    FISH_IMG = {
        'src': os.fspath("./fish_jpgs"),
        'dst': os.fspath("./fish_jpgs_rectified")
    }

def raw_to_png(img_type: ImageType):
    # rawData = open(path, 'rb').read()
    # imgSize = (3000,4000) # the image size
    # img = Image.frombytes('RGB', imgSize, rawData)
    # img.save("foo.png")# can give any format you like .png
    pass
    
if __name__ == "__main__":
    calib_file = os.fspath('calibration_data.tar')
    calibration_mat, distortion_coeffs = read_camera_calibration(calib_file)
    
    for image_type in (ImageType):
        src_path = image_type['src'] 
        dst_path = image_type['dst']
        filenames = glob.glob(os.path.join(src_path,'*.JPG'))
        for filename in filenames:
            img = cv2.imread(filename)
            undistorted_img = cv2.undistort(img, calibration_mat, distortion_coeffs)
            cv2.imwrite(undistorted_img, os.path.join(dst_path, os.path.basename(filename)))
            
