from camera_imaging_pipeline.src.image_processing import imageProcessing
import os
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

#read a sample image, read the parametrs .json file and load the parameters as a dictionary
#img1_path = os.path.join(os.path.dirname(__file__), '.\data\\test_data\Fishsense.dng')
#img1_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/7_23_nathans_pool/FSL-01D Fred/FSL-01D Fred/P7130166.ORF")     
img2_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/7_23_La_Jolla_Kelp_Beds/Red_Laser_Test/P7170011.ORF") 

json_path_1 = os.path.join(os.path.dirname(__file__), r'.\\camera_imaging_pipeline\\params1.json')
json_path_2 = os.path.join(os.path.dirname(__file__), r'.\\camera_imaging_pipeline\\params2.json')
params1 = json.load(open(json_path_1))
params2 = json.load(open(json_path_2))


#create a processing pipeline based on the parameters dictionary in params1, apply that pipeline to an image
#and finally display that image. 
#config1 = imageProcessing()
config2 = imageProcessing()
#img1_data, img1_visual = config1.applyToImage(img1_path, params1)
img2_data, img2_visual = config2.applyToImage(img2_path, params1)
#img2_data, img2_visual = config1.applyToImage(img2_path)
#img3_data, img3_visual = config1.applyToImage(img3_path)
#config1.showImage(img1_visual)
cv2.imwrite("./data/ocean_test.png", img2_data)
config2.showImage(img2_visual)

#config1.showImage(img2_visual)
#config1.showImage(img3_visual)