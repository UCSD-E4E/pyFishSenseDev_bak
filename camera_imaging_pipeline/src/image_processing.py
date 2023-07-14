import numpy as np
import rawpy
import cv2 as cv
from utils.h_functions import *

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
        self.img = None

    #calls the relevant functions from h_functions.py for the processing pipeline,
    #according to the parameters specified in params1.json.
    #The method returns the image both in it's original size for further processing, 
    #and it returns the resized data for displaying the processed image conveniently to 
    #the user.
    def applyToImage(self, img_path):
        self.raw = rawpy.imread(img_path)
        self.img = self.raw.raw_image.copy()

        if self.processes['linearization'] == True:
            self.img = linearization(self.img)
        if self.processes['demosaic'] == True:
            self.img = demosaic(self.img)
        if self.processes['denoising'] == True:
            self.img = denoising(self.img, self.denoising_val)
        if self.processes['colorSpace'] == True:
            self.img = colorSpace(self.img, self.colour)
        if self.processes['exposureComp'] == True:
            self.img = exposureComp(self.img, self.exposure_val)
        if self.processes['toneCurve'] == True:
            self.img = toneCurve(self.img, self.tone_curve)
        if self.processes['gammaCorrection'] == True:
            self.img = gammaCorrection(self.img, self.gamma_correction)
        if self.processes['greyWorldWB'] == True:
            self.img = greyWorldWB(self.img, self.colour)
        
        return self.img ,imageResize(self.img, self.resize_val)

    #method for returning the processed image.
    def getImage(self):
        if self.img.all() == None: 
            print('No image loaded')
        else:
            return self.img

    #method for displaying the processed image to the user. 
    def showImage(self, img):
        if self.img.all() == None:
            print('No image loaded')
        else:
            cv.imshow("urer", img)
            k = cv.waitKey(0)
            cv.destroyAllWindows()