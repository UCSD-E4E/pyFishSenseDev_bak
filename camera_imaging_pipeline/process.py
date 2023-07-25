from glob import glob
from src.image_processing import imageProcessing
import json
import os
import cv2

def main():
    json_path = os.path.join(os.path.dirname(__file__), 'params1.json')
    params1 = json.load(open(json_path))

    raw_images = glob("./data/**/*.ORF", recursive=True)

    for raw in raw_images:
        config1 = imageProcessing(params1)
        img1_data, _ = config1.applyToImage(raw)
        
        new_path = raw.replace("raw", "png").replace(".ORF", ".PNG")
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        cv2.imwrite(new_path, img1_data)

    print(raw_images)

if __name__ == '__main__':
    main()
