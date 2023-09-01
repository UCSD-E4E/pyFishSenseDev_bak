from .src.image_processing import imageProcessing
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

#read a sample image, read the parametrs .json file and load the parameters as a dictionary
# img_path = os.path.join(os.path.dirname(__file__), '.\data\Fishsense.dng')
img_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/camera_imaging_pipeline/data\P7170124.ORF")
json_path = os.path.join(os.path.dirname(__file__), '.\params1.json')
params1 = json.load(open(json_path))

#create a processing pipeline based on the parameters dictionary in params1, apply that pipeline to an image
#and finally display that image. 
config1 = imageProcessing(params1)
img1_data, img1_visual = config1.applyToImage(img_path.as_posix())

cv2.imwrite('raw_img.png', img1_data)

# cv2.imshow('processed image', img1_visual)
# cv2.waitKey(0)
# cv2.destroyAllWindows()