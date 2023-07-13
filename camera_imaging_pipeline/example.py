from src.image_processing import imageProcessing
import os
import json

#read a sample image, read the parametrs .json file and load the parameters as a dictionary
img_path = os.path.join(os.path.dirname(__file__), '.\data\Fishsense.dng')
json_path = os.path.join(os.path.dirname(__file__), '.\params1.json')
params1 = json.load(open(json_path))

#create a processing pipeline based on the parameters dictionary in params1, apply that pipeline to an image
#and finally display that image. 
config1 = imageProcessing(params1)
img1_data, img1_visual = config1.applyToImage(img_path)
config1.showImage(img1_visual)