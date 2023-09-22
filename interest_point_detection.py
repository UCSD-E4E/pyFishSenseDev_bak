###################################################################
####### This is an environment for testing the SIFT algorithm #####
###################################################################

import cv2
import numpy as np
from pathlib import Path
from camera_imaging_pipeline.src.image_processing import imageProcessing




def display_interest_points_matrix(img: np.ndarray) -> np.ndarray:

    img_copy = img.copy()
    sift = cv2.SIFT_create(40, 5, 0.001, 100, 0.8)
    kp = sift.detect(img)
    img_copy = cv2.drawKeypoints(img_copy, kp, img_copy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return img_copy

# def display_interest_points(laser_path: Path, 
#                          calibration_path:Path, filepath: Path, vanishing_point: np.ndarray, param_path=None):
    

#     print(filepath.suffix)

#     if filepath.suffix == ".ORF":
#         params = json.load(open(params_path))
#         processor = imageProcessing()
#         img, _ = processor.applyToImage(filepath, params)

#     else:
#         img = cv2.imread(filepath.as_posix())

#     _, mask = get_masked_image_matrix(laser_path, calibration_path, img)
#     mask = mask.astype('uint8')
#     # img = cv2.imread('P7170114.JPG')
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#     img_copy = img.copy()
#     #masked_copy = masked_image.copy()
#     #print('Contrast Threshold: ' +str(i))
#     sift = cv2.SIFT_create(20, 3, 0.14, 5, 2.1)
#     kp = sift.detect(img_copy, mask)
#     print(cv2.KeyPoint_convert(kp))
#     print(len(kp))

#     ones = np.ones_like(img_copy) * 255
#     img_copy = cv2.drawKeypoints(img_copy, kp, img_copy, flags=4)
#     ones = cv2.drawKeypoints(ones, kp, ones, flags=4)
#     #img_copy = cv2.circle(img_copy, kp, 5, 5, 20)

#     # resized = cv2.resize(img, (1200, 750))
#     # cv2.imshow("Detected Keypoints", resized)
#     # k = cv2.waitKey(0)

    
#     resized = cv2.resize(img_copy, (1200, 750))
#     resized_ones = cv2.resize(ones, (1200, 750))
#     cv2.imshow("resized", zoom_at(resized, 2.2, coord=(1200/2, 750/2)))
#     cv2.imshow("zeros", zoom_at(resized_ones, 3, coord=(1200/2, 750/2)))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# # laser_path_old = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/old/laser-calibration-output-4-12-bot-float.dat")
# # calibration_path_old = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/old/calibration-output.dat")    
# laser_path_new = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/laser-calibration-output-7-13.dat")
# calibration_path_new = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/fsl-01d-lens.dat")
# data_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/7_23_La_Jolla_Kelp_Beds/Safety_Stop_Red")
# laser_data_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/laser_templates/laser.png")
# params_path = Path('C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/camera_imaging_pipeline/params1.json')





# display_interest_points(laser_path_new, calibration_path_new, laser_data_path, None, params_path)