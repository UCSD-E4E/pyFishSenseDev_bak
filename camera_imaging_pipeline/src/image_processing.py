from pathlib import Path

import cv2 as cv
import numpy as np
import rawpy

from ..utils.processing_functions import *


#this class allows the user to configure a image processing pipeline by adjusting
#the paramters in the params1.json file. Creat a new .json for each desired 
# configuration.
class imageProcessing():

    #calls the relevant functions from h_functions.py for the processing pipeline,
    #according to the parameters specified in params1.json.
    #The method returns the image both in it's original size for further processing, 
    #and it returns the resized data for displaying the processed image conveniently to 
    #the user.
    def applyToImage(self, img_path, params):

        resize_val = params['resize_val']
        gamma_correction = params['gamma_correction']
        tone_curve = params['tone_curve']
        exposure_val = params['exposure_compensation']
        colour = params['colour']
        denoising_val = params['denoising']
        processes = params['processes']
        if type(img_path) != np.ndarray:
            
            img = rawpy.imread(img_path.as_posix()).raw_image.copy()

            if processes['linearization'] == True:
                print("Linearizing")
                img = linearization(img)
            if processes['demosaic'] == True:
                print("Demosaicing")
                img = demosaic(img)
            if processes['denoising'] == True:
                print("Denoising")
                img = denoising(img, denoising_val)
            if processes['colorSpace'] == True:
                print("Colorspacing")
                img = colorSpace(img, colour)
            if processes['exposureComp'] == True:
                print("Exposure Compositioning")
                img = exposureComp(img, exposure_val)
            if processes['toneCurve'] == True:
                print("Tone Curving")
                img = toneCurve(img, tone_curve)
            if processes['gammaCorrection'] == True:
                print("Gamma Correcting")
                img = gammaCorrection(img, gamma_correction)
            if processes['greyWorldWB'] == True:
                print("Whitepoint Correcting")
                img = greyWorldWB(img, colour)
            
            #img = scale_pixels(img)
            img = (img/256).astype('uint8')
            print(img.max())
            return img ,imageResize(img, resize_val)

        else: 
            img = img_path

            if processes['linearization'] == True:
                img = linearization(img)
            if processes['demosaic'] == True:
                img = demosaic(img)
            if processes['denoising'] == True:
                img = denoising(img, denoising_val)
            if processes['colorSpace'] == True:
                img = colorSpace(img, colour)
            if processes['exposureComp'] == True:
                img = exposureComp(img, exposure_val)
            if processes['toneCurve'] == True:
                img = toneCurve(img, tone_curve)
            if processes['gammaCorrection'] == True:
                img = gammaCorrection(img, gamma_correction)
            if processes['greyWorldWB'] == True:
                img = greyWorldWB(img, colour)
            # if processes['contrastStretching'] == True:
            #     img = contrastStretching(img)
            
            #img = scale_pixels(img)
            #img = (img/256).astype('uint8')
            return img ,imageResize(img, resize_val)

    # #method for returning the processed image.
    # def getImage(self):
    #     if img.all() == None: 
    #         print('No image loaded')
    #     else:
    #         return img

    #method for displaying the processed image to the user. 
    def showImage(self, img):
        if img.all() == None:
            print('No image loaded')
        else:
            cv.imshow("urer", img)
            k = cv.waitKey(0)
            cv.destroyAllWindows()