import numpy as np
import rawpy
import matplotlib.pyplot as plt
import cv2 as cv
import os
import json
from utils.h_functions import *


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
            self.img = greyWorldWB(self.img)
        
        return self.img ,imageResize(self.img, self.resize_val)

    def getImage(self):
        if self.img.all() == None: 
            print('No image loaded')
        else:
            return self.img

    def showImage(self, img):
        if self.img.all() == None:
            print('No image loaded')
        else:
            cv.imshow("urer", img)
            k = cv.waitKey(0)
            cv.destroyAllWindows()