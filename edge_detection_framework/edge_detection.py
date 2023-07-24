import cv2
import scipy.fft as ft
from scipy import datasets
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
from helpers import cropImage, scale_data

def edgeDetection(img, rad, sigma, roi, channel=None):
    if type(img) != np.ndarray:
        img = plt.imread(img_path.as_posix()).astype('float16')

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
img_path = Path("C:Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/edge_detection_framework/test_data/test_fred.png")
img = plt.imread(img_path.as_posix()).astype('float16')


# cut = mask2[int(np.floor(mask2.shape[0]/2)), :]
# x = np.linspace(0, len(cut), len(cut))

roi = [0, 100, 800, 800]
_, img_f = edgeDetection(img_path, 60, 0, roi, channel='r')

label = "Filtered Image"
fig, axs = plt.subplots(1,2)
plt.title(label)
axs[0].imshow(img, cmap='gray')
axs[1].imshow(img_f, cmap='gray')
plt.show()