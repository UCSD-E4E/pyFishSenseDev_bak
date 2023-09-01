import cv2
import numpy as np
from pathlib import Path
from edge_detection_framework.helpers import cropImage, scale_data
import rawpy
import json
#from detect_laser import get_masked_image_matrix, detect_laser_raw
import glob
from camera_imaging_pipeline.src.image_processing import imageProcessing
import json
from scipy import signal

def edgeDetection(img, rad, sigma, roi, channel=None):
    #if a string is passed to the function, and the file is anything other than a .ORF, such as a .jpg, then the image is read in with the opencv imread() function
    if type(img) != np.ndarray and img.suffix != ".ORF":
        img = cv2.imread(img.as_posix()).astype('float16')
    # if img.dtype != np.float16:
    #     img = img.astype('float16')

    #if a string is passed to the function, and the image is a .ORF raw file, then the we read in the image with rawpy and convert it to a numpy array.
    if type(img) != np.ndarray and img.suffix == ".ORF":
        img = np.asarray(rawpy.imread(img.as_posix()).raw_image).astype('float16')

    # if no value is passed as the region of interest, then we automatically take the whole image. 
    if roi == None:
        roi = [0, 0, img.shape[1], img.shape[0]]
    
    # the desired channel is selected depending on the keyword passed to the channel= keyword argument. 
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
    #b_notch, a_notch = signal.iirnotch(4.61*pow(10, 14), 30, 4.61*pow(10, 14)*2)
    #notch_filter= b_notch/a_notch
    dft_shift_masked2 = np.multiply(dft_shift, mask2) / 255

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
laser_p = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/laser-calibration-output-7-13.dat")
cal_p = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/fsl-01d-lens.dat")

params_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/camera_imaging_pipeline/params1.json")
img_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/7_23_La_Jolla_Kelp_Beds/Safety_Stop_Red")
#img_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/7_23_nathans_pools/FSL-01F_Fred")
files = list(img_path.glob("*.ORF"))

# config1 = imageProcessing()

# with open(params_path, 'r', encoding='ascii') as handle:
#     params = json.load(handle)

# for img_path in files[4:]:
#     img, _ = config1.applyToImage(img_path, params)
#     img = scale_data(img, 8)
#     # print(img_path.name)


#     _, img_filtered = edgeDetection(img, 130, 3, None, channel='gray')

#     #cv2.namedWindow("Masked_Window", cv2.WINDOW_NORMAL)
#     resized = cv2.resize(img_filtered, (1200, 750))


#     # cv2.imshow("Filtered Image", resized)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     # save_path = Path("./data/filtered") / str(img_path.name)
#     # save_path.with_suffix(".jpg")
#     # print(save_path)
#     # cv2.imwrite(save_path.as_posix(), resized_f)