from src.image_processing import imageProcessing
import os
import json


img_path = os.path.join(os.path.dirname(__file__), '.\data\Fishsense.dng')
json_path = os.path.join(os.path.dirname(__file__), '.\params1.json')
params1 = json.load(open(json_path))

config1 = imageProcessing(params1)
img1_data, img1_visual = config1.applyToImage(img_path)
config1.showImage(img1_visual)