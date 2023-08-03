import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy


def intensity_distrobution(img_path):
        raw = rawpy.imread(img_path)
        img = raw.raw_image.copy()
        img_data = img.flatten().astype('uint8')

        intensities = np.zeros((255,2))
        intensities[:,0] =  np.linspace(0,255,num=255).astype('uint8')
        
        for i in range(255):
            intensities[i,1] = len(img_data[img_data == i])

        return intensities

        
    


img1_path = os.path.join(os.path.dirname(__file__), r'.\data\\7_23_nathans_pool\\TG6_lens_calibration\\P7130005.ORF')

histogram = intensity_distrobution(img1_path)
plt.bar(histogram[:,0], histogram[:,1])

plt.show()