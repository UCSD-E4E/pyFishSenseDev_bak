import numpy as np
import cv2
import matplotlib.pyplot as plt


def closest_to_vanishing_point(point_list, vanishing_point):
    if len(point_list[0]) > 0:
        min_dist = 10000
        for i in range(len(point_list[0])):
            length = np.sqrt(np.square(point_list[i][0] - vanishing_point[0]) + np.square(point_list[i][1] - vanishing_point[1]))
            if length < min_dist:
                min_dist = length
                min_dist_point = (point_list[i][0], point_list[i][1])

    return min_dist_point


def get_optimal_keypoint(color_image, kp_list, vanishing_point):
    point_list = cv2.KeyPoint_convert(kp_list)
    print(len(point_list))
    if len(point_list) > 1:
        print('Hello')
        optimal_kp_point = closest_to_vanishing_point(point_list, vanishing_point)
        optimal_kp = cv2.KeyPoint_convert([np.asarray(optimal_kp_point)])
    
    else:
        optimal_kp = kp_list
    
    return optimal_kp


def get_redest_keypoint(color_img, kp_list):
    print(kp_list)
    if kp_list != ():
        point_list = cv2.KeyPoint_convert(kp_list).astype('uint16')
        #hsv = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)
        max_red_val = 0
        optimal_kp = []

        for point in point_list:
            print(point)
            r = 10
            y = point[0]
            x = point[1]
            area = color_img[(x-r):(x+r), (y-r):(y+r)]

            mean_red_val = np.mean(area[:,:,0])
            if mean_red_val > max_red_val:
                max_red_val = mean_red_val
                optimal_kp = point

        optimal_keypoint_out = cv2.KeyPoint_convert([optimal_kp])
        print("This is the optimal keypoint: ")
        print(optimal_kp)

    else:
        optimal_keypoint_out = kp_list
        print('No keypoints detected!')
        
    return optimal_keypoint_out



def get_hsv_mask(color_img):

    img_hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))

    ## Merge the mask and crop the red regions
    mask = cv2.bitwise_or(mask1, mask2)
    croped = cv2.bitwise_and(color_img, color_img, mask=mask)

    return mask




