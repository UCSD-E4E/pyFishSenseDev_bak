from pathlib import Path
import cv2
import os
import sys
import numpy as np
import rawpy
import json
from helpers.img_zoom import zoom_at
from helpers.closest_to_vanishing_point import closest_to_vanishing_point, get_optimal_keypoint, get_redest_keypoint, get_hsv_mask
import matplotlib.pyplot as plt

from camera_imaging_pipeline.src.image_processing import imageProcessing

from detect_laser_line import return_line, get_vanishing_point_2d
from typing import List, Tuple

#from edge_detection import edgeDetection


def get_masked_image_with_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:

    mask_rgb = np.dstack((mask,mask,mask)).astype(np.uint8)
    masked_image = img * mask_rgb
    return masked_image

def return_mask_stack(mask_list: List[np.ndarray]) -> np.ndarray:

    output = np.zeros(mask_list[0].shape)

    for mat in mask_list:
        output = np.logical_or(output, mat).astype(np.uint8)
    
    return output
    

def get_vanishing_point(laser_path: Path, calibration_path: Path):

    return get_vanishing_point_2d(laser_path, calibration_path)

def get_mask(laser_path: Path, calibration_path: Path, shape: Tuple[int, int]) -> np.ndarray:

    mask = np.zeros(shape=shape)
    start_point, end_point = return_line(laser_path, calibration_path)
    mask = cv2.line(mask, start_point, end_point, color=1, thickness=75)

    mask = mask.astype(np.uint8)
    return mask

def get_masked_image_matrix(laser_path: Path, 
                         calibration_path:Path, img: np.ndarray):
    
    # Make sure that the file is a JPG
    if type(img) != np.ndarray:
        sys.exit(1)

    if len(np.shape(img)) == 3:
        #print('3')
        # Get the line
        img_clone = img.copy()
        start_point, end_point = return_line(laser_path, calibration_path)
        cv2.line(img_clone, start_point, end_point, color=(0,0,255), thickness=75) 

        # Create the mask and the masked image
        mask = cv2.inRange(img_clone, (0,0,255), (0,0,256))/255
        mask_rgb = np.dstack((mask,mask,mask)).astype(np.uint8)
        masked_image = img * mask_rgb

        return masked_image, mask

    elif (len(np.shape(img)) == 2):
        print(np.max(img))
        img_clone = img.copy()
        start_point, end_point = return_line(laser_path, calibration_path)
        cv2.line(img_clone, start_point, end_point, color=0, thickness=75)

        mask = cv2.inRange(img_clone, 0, 1)/255
        mask = mask.astype(np.uint8)
        masked_image = img * mask

        return masked_image, mask
    sys.exit(1)

def get_masked_image_rgb(laser_path: Path, 
                         calibration_path:Path, filepath: Path):
    
    # Make sure that the file is a JPG
    if filepath.suffix != ".JPG":
        sys.exit(1)
    
    img = cv2.imread(filepath.as_posix())
    img_clone = img.copy()

    # Get the line
    start_point, end_point = return_line(laser_path, calibration_path)
    cv2.line(img_clone, start_point, end_point, color=(0,0,255), thickness=75) 

    # Create the mask and the masked image
    mask = cv2.inRange(img_clone, (0,0,255), (0,0,256))/255
    mask_rgb = np.dstack((mask,mask,mask)).astype(np.uint8)
    masked_image = img * mask_rgb

    return masked_image

def get_masked_image_raw(laser_path: Path, 
                         calibration_path:Path, filepath: Path):
    
    # Make sure that the file is a JPG
    if filepath.suffix != ".ORF":
        sys.exit(1)
    
    with rawpy.imread(filepath.as_posix()) as raw:

        raw_img = raw.raw_image.copy()
        
        img = (raw_img/np.amax(raw_img)*255).astype(np.uint8)
        
        img_clone = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Get the line
        start_point, end_point = return_line(laser_path, calibration_path)
        cv2.line(img_clone, start_point, end_point, color=(0,0,255), thickness=75) 

        # Create the mask and the masked image
        mask = (cv2.inRange(img_clone, (0,0,255), (0,0,256))/255).astype(np.uint8)
        masked_image = img * mask

        return masked_image
    
def detect_laser_raw(masked_raw: np.ndarray, vanishing_point: np.array):
    
    contour_mask = cv2.inRange(masked_raw, 250, 256)
    all_pts = np.nonzero(contour_mask)
    pt_min_dist = []
    
    # if len(all_pts[0]) > 0:
    #     min_dist = 10000
    #     for i in range(len(all_pts[0])):
    #         length = np.sqrt(np.square(all_pts[0][i] - vanishing_point[0]) + np.square(all_pts[1][i] - vanishing_point[1]))
    #         if length < min_dist:
    #             min_dist = length
    #             pt_min_dist = (all_pts[0][i], all_pts[1][i])

                              
    #     contour_mask = np.zeros_like(contour_mask)
    #     contour_mask[pt_min_dist[0]][pt_min_dist[1]] = 255

    kernel = np.ones((5,5), np.uint8)
    contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(contour_mask, 
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def display_masked_image(laser_path: Path, 
                         calibration_path:Path, filepath: Path, vanishing_point: np.array, raw: bool, param_path=None):
    
    if not raw:
        masked_image = get_masked_image_rgb(laser_path, calibration_path, filepath)
        cv2.namedWindow("Masked_Window", cv2.WINDOW_NORMAL)
        cv2.imshow("Masked_Window",masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        #masked_image = get_masked_image_matrix(laser_path, calibration_path, filepath)
        masked_image = get_masked_image_raw(laser_path, calibration_path, filepath)

        contours = detect_laser_raw(masked_image, vanishing_point)

        params = json.load(open(params_path))
        processor = imageProcessing()
        processed_image, _ = processor.applyToImage(masked_image, params)

        for cnt in contours:
            cv2.drawContours(processed_image,[cnt],0,(0,0,65535),thickness=25)
        cv2.namedWindow("Masked_Window", cv2.WINDOW_NORMAL)
        #cv2.imshow("Masked_Window",masked_rgb)
        # resized = cv2.resize(masked_rgb, (1800, 1350))
        resized = cv2.resize(processed_image, (1200, 750))
        #cv2.imshow("resized", masked_rgb[1000:2000, 1000:3000])
        cv2.imshow("resized", resized)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def display_detection(laser_path: Path, 
                         calibration_path:Path, filepath: Path, vanishing_point: np.array, params_path: Path):
        
        params = json.load(open(params_path))
        processor = imageProcessing()
        processed_image, _ = processor.applyToImage(filepath, params)
        #processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)

        masked_image = get_masked_image_matrix(laser_path, calibration_path, processed_image)
        #masked_image = get_masked_image_raw(laser_path, calibration_path, processed_image)

        contours = detect_laser_raw(masked_image, vanishing_point)   

        for cnt in contours:
            cv2.drawContours(masked_image,[cnt],0,(0,0,65535),thickness=-1)
        
        resized = cv2.resize(processed_image, (1200, 750))
        cv2.imshow("resized", resized)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# def display_filtered_image(laser_path: Path, 
#                         calibration_path:Path, filepath: Path, vanishing_point: np.array, param_path=None):

#     params = json.load(open(params_path))
#     processor = imageProcessing()
#     processed_image, _ = processor.applyToImage(filepath, params)
#     _, filtered_image = edgeDetection(processed_image, 150, 0, None, channel='r')

#     #masked_image = get_masked_image_matrix(laser_path, calibration_path, filtered_image)

#     contours = detect_laser_raw(filtered_image, vanishing_point)   

#     img_clone = filtered_image.copy()
#     img_clone = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2RGB)


#     for cnt in contours:
#         cv2.drawContours(img_clone,[cnt],-1,(0,0,65535),thickness=8)
    
#     resized_original = cv2.resize(processed_image, (1200, 750))
#     resized_filtered = cv2.resize(img_clone, (1200, 750))
#     cv2.imshow("original resized", resized_original)
#     cv2.imshow("resized", resized_filtered)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    

# def display_filtered_masked_image(laser_path: Path, 
#                          calibration_path:Path, filepath: Path, vanishing_point: np.array, param_path=None):
    
#     params = json.load(open(params_path))
#     processor = imageProcessing()
#     processed_image, _ = processor.applyToImage(filepath, params)
#     _, filtered_image = edgeDetection(processed_image, 300, 0, None, channel='r')

#     masked_image = get_masked_image_matrix(laser_path, calibration_path, filtered_image)

#     contours = detect_laser_raw(masked_image, vanishing_point)   

#     img_clone = filtered_image.copy()
#     img_clone = cv2.cvtColor(img_clone, cv2.COLOR_GRAY2RGB)


#     for cnt in contours:
#         cv2.drawContours(img_clone,[cnt],-1,(0,0,65535),thickness=10)
    

#     resized = cv2.resize(img_clone, (1200, 750))
#     cv2.imshow("resized", resized)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def display_interest_points(laser_path: Path, 
#                          calibration_path:Path, filepath: Path, vanishing_point: np.array, param_path_red=None, params_path_color=None):
    
#     params = json.load(open(params_path_red))
#     processor = imageProcessing()
#     img_red, _ = processor.applyToImage(filepath, params)

#     params = json.load(open(params_path_color))
#     processor = imageProcessing()
#     img_color, _ = processor.applyToImage(filepath, params)

#     _, mask = get_masked_image_matrix(laser_path, calibration_path, img_color)
#     mask = mask.astype('uint8')

#     img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)


#     img_color_copy = img_color.copy()
#     img_red_copy = img_red.copy()

#     sift = cv2.SIFT_create(20, 3, 0.08, 10, 1.6)
#     kp = sift.detect(img_red_copy, mask)

#     #optimal_kp = get_optimal_keypoint(img_color_copy, kp, vanishing_point)
#     optimal_keypoint = get_redest_keypoint(img_color_copy, kp)
#     #masked = get_hsv_mask(img_color_copy)

#     ones = np.ones_like(img_red_copy) * 255
#     img_out = cv2.drawKeypoints(img_color_copy, optimal_keypoint, img_color_copy, flags=4)
#     ones = cv2.drawKeypoints(ones, kp, ones, flags=4)
    
#     # resized = cv2.resize(img_copy, (1200, 750))
#     # resized_ones = cv2.resize(ones, (1200, 750))
#     # cv2.imshow("resized", zoom_at(resized, 2.2, coord=(1200/2, 750/2)))
#     # cv2.imshow("zeros", zoom_at(resized_ones, 3, coord=(1200/2, 750/2)))
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     plt.imshow(img_out)
#     plt.show()


# laser_path_old = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/old/laser-calibration-output-4-12-bot-float.dat")
# calibration_path_old = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/old/calibration-output.dat")    
# laser_path_new = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/laser-calibration-output-7-13.dat")
# calibration_path_new = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/fsl-01d-lens.dat")
# data_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/7_23_La_Jolla_Kelp_Beds/Safety_Stop_Red")
# vanishing_point = get_vanishing_point(laser_path_new, calibration_path_new)[0:2]


# for file in os.listdir(data_path.as_posix())[20:]:

#     filepath = data_path.joinpath(file)
#     if filepath.suffix != ".ORF":
#         continue
    
#     params_path_red = Path('C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/camera_imaging_pipeline/params_red.json')
#     params_path_color = Path('C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/camera_imaging_pipeline/params_color.json')

#     print(f"Processing {file}")
#     # display_detection(laser_path, calibration_path, filepath, params_path)
#     #display_detection(laser_path_old, calibration_path_old, filepath, vanishing_point, params_path)
#     #display_masked_image(laser_path, calibration_path, filepath, vanishing_point, True, params_path)
#     #display_filtered_image(laser_path_new, calibration_path_new, filepath, vanishing_point, params_path)
#     display_interest_points(laser_path_new, calibration_path_new, filepath, vanishing_point, params_path_red, params_path_color)