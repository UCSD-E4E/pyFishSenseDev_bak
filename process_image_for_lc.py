# Purpose of this file is to process images for lense calibration tool

from pathlib import Path
import os
import json
import cv2
import rawpy

from camera_imaging_pipeline.src.image_processing import imageProcessing

# path for data

data_path = Path('data/Lens calibration with corrective optic')
json_path = Path('camera_imaging_pipeline/params1.json')
params = json.load(open(json_path))
processor = imageProcessing(params)

destination_path = data_path.joinpath('processed_images/')
destination_path.mkdir(exist_ok=True)

for file in os.listdir(data_path):
    print(f"Processing {file}")
    filepath = data_path.joinpath(file)
    if filepath.suffix != ".ORF":
        continue
    processed_img, show = processor.applyToImage(filepath.as_posix())

    cv2.imshow("nothing", show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# # path for data

# data_path = Path('/home/viva/Fishsense/fishsense-lite-python-pipeline/data/TG6 Lens Calibration')
# json_path = Path('camera_imaging_pipeline/params1.json')
# params = json.load(open(json_path))
# processor = imageProcessing(params)

# destination_path = data_path.joinpath('processed_images/')
# destination_path.mkdir(exist_ok=True)

# for file in os.listdir(data_path):
#     print(f"Processing {file}")
#     filepath = data_path.joinpath(file)
#     if filepath.suffix != ".ORF":
#         continue
#     processed_img, show = processor.applyToImage(filepath.as_posix())

#     cv2.imshow("nothing", show)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()