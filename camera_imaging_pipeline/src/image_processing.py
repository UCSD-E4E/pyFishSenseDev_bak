import numpy as np
import rawpy
import cv2 as cv
from utils.h_functions import *
import gc

#this class allows the user to configure a image processing pipeline by adjusting
#the paramters in the params1.json file. Creat a new .json for each desired 
# configuration.
class imageProcessing():
    def __init__(self, params):
        self.resize_val = params['resize_val']
        self.gamma_correction = params['gamma_correction']
        self.tone_curve = params['tone_curve']
        self.exposure_val = params['exposure_compensation']
        self.colour = params['colour']
        self.denoising_val = params['denoising']
        self.processes = params['processes']

    #calls the relevant functions from h_functions.py for the processing pipeline,
    #according to the parameters specified in params1.json.
    #The method returns the image both in it's original size for further processing, 
    #and it returns the resized data for displaying the processed image conveniently to 
    #the user.
    def applyToImage(self, img_path):
        img = None

        while img is None or np.average(img) < 2000:
            # if img is not None:
            #     print("Repeating...")

            with rawpy.imread(img_path) as raw:
                img = raw.raw_image.copy()

            img = img[:, :-53]

            if self.processes['linearization'] == True:
                img = linearization(img)

        if self.processes['demosaic'] == True:
            img = demosaic(img)
        if self.processes['denoising'] == True:
            img = denoising(img, self.denoising_val)
        if self.processes['colorSpace'] == True:
            img = colorSpace(img, self.colour)
        if self.processes['exposureComp'] == True:
            img = exposureComp(img, self.exposure_val)
        if self.processes['toneCurve'] == True:
            img = toneCurve(img, self.tone_curve)
        if self.processes['gammaCorrection'] == True:
            img = gammaCorrection(img, self.gamma_correction)
        if self.processes['greyWorldWB'] == True:
            img = greyWorldWB(img, self.colour)

        img = equalizeHist(img)
        # img = cv.convertScaleAbs(img)
        return img, imageResize(img, self.resize_val)