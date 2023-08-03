from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from matplotlib import cbook, cm
from matplotlib.colors import LightSource

from edge_detection_framework.helpers import cropImage, scale_data

#from detect_laser import get_masked_image_matrix, detect_laser_raw
#import glob
#from camera_imaging_pipeline.src.image_processing import imageProcessing
#import json

def edgeDetection(img, rad, sigma, roi, channel=None):
    if type(img) != np.ndarray and img.suffix != ".ORF":
        img = plt.imread(img.as_posix()).astype('float16')
    # if img.dtype != np.float16:
    #     img = img.astype('float16')

    if type(img) != np.ndarray and img.suffix == ".ORF":
        img = np.asarray(rawpy.imread(img.as_posix()).raw_image).astype('float16')

    if roi == None:
        roi = [0, 0, img.shape[1], img.shape[0]]
    
    if len(img.shape) == 3:
        if channel == 'r':
            img = img[:,:,0]
        elif channel == 'g':
            img = img[:,:,1]
        elif channel == 'b':
            img = img[:,:,1]
        elif channel == 'gray':
            img = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3
    #Image pre-processing: select red channel, crop to ROI, scale the data
      
    #img = img[:,:,0]
    img = cropImage(img, roi)
    img = scale_data(img, 8)

    #Apply the fft and shift the zero frequency component to the center of the array
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    #Create a mask for a ILPF, subtract it from 255 to create the IHPF. Higher radius values result in more information being removed from the image,
    #with only the very high frequencies remaining
    radius = rad
    mask = np.zeros_like(img)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]
    mask = 255 - mask

    #Apply gaussian blur to create a GHPF. Then multiply the shifted DFT with the GHPF.
    mask2 = cv2.GaussianBlur(mask, (19,19), sigma)
    dft_shift_masked2 = np.multiply(dft_shift,mask2) / 255

    #Shift back the DFT and the filter to be able to display the images correctly. 
    back_ishift = np.fft.ifftshift(dft_shift)
    back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)

    #Then compute the IDFT of the origianl image and the filtered and shifted image. 
    img_back = np.fft.ifft2(back_ishift, axes=(0,1))
    img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

    img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
    img_filtered2 = np.abs(10*img_filtered2).clip(0,255).astype(np.uint8)

    return img_back, img_filtered2


#img_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/img_filtering/P7130166.JPG")
# laser_p = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/laser-calibration-output-7-13.dat")
# cal_p = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/fsl-01d-lens.dat")

# params_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/camera_imaging_pipeline/params1.json")
# img_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/7_23_La_Jolla_Kelp_Beds/Red_Laser_Test")
# img_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/7_23_nathans_pools/FSL-01D Fred")
# files = list(img_path.glob('*.ORF'))

# config1 = imageProcessing()
# with open(params_path, 'r', encoding='ascii') as handle:
#     params = json.load(handle)

# for img_path in files:
#     img, _ = config1.applyToImage(img_path, params)
#     img = scale_data(img, 8)

#     roi = [0, 0, img.shape[1], img.shape[0]]
#     _, img_filtered = edgeDetection(img, 170, 3, roi, channel='r')

#     img_f = get_masked_image_matrix(laser_p, cal_p, img_filtered)
#     contours = detect_laser_raw(img_f)
#     img_clone = img_f.copy()
#     img_clone = cv2.cvtColor(img_clone, cv2.COLOR_GRAY2RGB)

#     for cnt in contours:
#         cv2.drawContours(img_clone, [cnt],0,(255,0,0),thickness=-1)

#     #if (about-to-crash()):
#     #   dont()

#     label = "Filtered Image"
#     fig, axs = plt.subplots(1,2)
#     plt.title(label)
#     axs[0].imshow(img, cmap='gray')
#     axs[1].imshow(img_clone, cmap='gray')
#     plt.show()

    # cv2.namedWindow("Masked_Window", cv2.WINDOW_NORMAL)
    # cv2.imshow("Masked_Window", img_clone)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()