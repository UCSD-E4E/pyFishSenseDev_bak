import os
import json
import cv2
import rawpy
from pathlib import Path
import glob
from tqdm import tqdm

from camera_imaging_pipeline.src.image_processing import imageProcessing



# json_path_1 = os.path.join(os.path.dirname(__file__), r'.\\camera_imaging_pipeline\\params1.json')
# params1 = json.load(open(json_path_1))

# params1 = json.load(open(json_path_1))
# config1 = imageProcessing(params1)
# img1_data, img1_visual = config1.applyToImage(img1_path)



# jpg_list = glob.glob(os.fspath('data/laser_jpgs_rectified/*.JPG'))


data_path = Path('data//7_23_nathans_pool/FSL-01F_Fred')
json_path = Path('camera_imaging_pipeline/params1.json')
params = json.load(open(json_path))
processor = imageProcessing(params)

destination_path = data_path.joinpath('processed_images/')
destination_path.mkdir(exist_ok=True)

for file in tqdm(os.listdir(data_path)):
    #print(f"Processing {file}")
    filepath = data_path.joinpath(file)
    if filepath.suffix != ".ORF":
        continue
    processed_img, _ = processor.applyToImage(filepath.as_posix())
    destination_file = destination_path.joinpath(f'{filepath.stem}.png')
    
    cv2.imwrite(str(destination_file), processed_img)

