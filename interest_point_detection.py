###################################################################
####### This is an environment for testing the SIFT algorithm #####
###################################################################

import cv2
import numpy as np
import json 
from pathlib import Path
from camera_imaging_pipeline.src.image_processing import imageProcessing
from detect_laser import get_masked_image_matrix
from helpers.img_zoom import zoom_at




#this is a function for displaying an image with all the interest points found by the SIFT algorithm.
def display_interest_points(laser_path: Path, 
                         calibration_path:Path, filepath: Path, vanishing_point: np.array, param_path=None):
    

    print(filepath.suffix)

    # if the image is a string with the path of the image file as a .ORF raw file, then apply image processing, otherwise read in the image with opencv. 
    if filepath.suffix == ".ORF":
        params = json.load(open(params_path))
        processor = imageProcessing()
        img, _ = processor.applyToImage(filepath, params)

    else:
        img = cv2.imread(filepath.as_posix())

    #apply a mask to the image that only returns the values along the line that the laser may be situated on.
    _, mask = get_masked_image_matrix(laser_path, calibration_path, img)
    mask = mask.astype('uint8')

    #make a copy of the image to work on. 
    img_copy = img.copy()

    #initialize and apply the SIFT algorithm to the image copy. 
    sift = cv2.SIFT_create(20, 3, 0.14, 5, 2.1)
    kp = sift.detect(img_copy, mask)

    #initialize a matrix in order to draw the keypoints onto a white background. 
    ones = np.ones_like(img_copy) * 255

    #draw the keypoints found by the SIFT algorithm onto the image and onto the white image. 
    img_copy = cv2.drawKeypoints(img_copy, kp, img_copy, flags=4)
    ones = cv2.drawKeypoints(ones, kp, ones, flags=4)

    #resize the image and the white matrix and display both matrices.     
    resized = cv2.resize(img_copy, (1200, 750))
    resized_ones = cv2.resize(ones, (1200, 750))
    cv2.imshow("resized", zoom_at(resized, 2.2, coord=(1200/2, 750/2)))
    cv2.imshow("zeros", zoom_at(resized_ones, 3, coord=(1200/2, 750/2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# laser_path_old = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/old/laser-calibration-output-4-12-bot-float.dat")
# calibration_path_old = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/old/calibration-output.dat")    
laser_path_new = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/laser-calibration-output-7-13.dat")
calibration_path_new = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/fsl-01d-lens.dat")
data_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/7_23_La_Jolla_Kelp_Beds/Safety_Stop_Red")
laser_data_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/laser_templates/laser.png")
params_path = Path('C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/camera_imaging_pipeline/params1.json')





display_interest_points(laser_path_new, calibration_path_new, laser_data_path, None, params_path)