from glob import glob
from src.image_processing import imageProcessing
import json
import os
import cv2
from multiprocessing import Pool, cpu_count

def process(raw: str):
    new_path = raw.replace("raw", "png").replace(".ORF", ".PNG")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    if os.path.exists(new_path):
        return
    
    json_path = os.path.join(os.path.dirname(__file__), 'params1.json')
    params1 = json.load(open(json_path))

    config1 = imageProcessing(params1)
    img1_data, _ = config1.applyToImage(raw)

    cv2.imwrite(new_path, img1_data)

def main():
    raw_images = glob("./data/**/*.ORF", recursive=True)
    Pool(cpu_count()).map(process, raw_images)

if __name__ == '__main__':
    main()
