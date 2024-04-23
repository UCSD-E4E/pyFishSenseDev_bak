from glob import glob
from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pyfishsense import FishHeadTailDetector, FishSegmentationInference, ImageRectifier, LaserDetector, RawProcessor, WorldPointHandler

dives = ["MolHITW", "MolPeLe", "ConchLed", "dive 1", "dive 3", "dive 4"]

laser_params = {
    "MolHITW": "./data/laser-params-dive-1.pkg",
    "MolPeLe": "./data/laser-params-dive-1.pkg",
    "ConchLed": "./data/laser-params-dive-3.pkg",
    "dive 1": "./data/laser-params-dive-1.pkg",
    "dive 3": "./data/laser-params-dive-3.pkg",
    "dive 4": "./data/laser-params-dive-4.pkg",
}

def uint16_2_double(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) / 65535

def uint16_2_uint8(img: np.ndarray) -> np.ndarray:
    return (uint16_2_double(img) * 255).astype(np.uint8)

def process(input_file: str):
    dive = [d for d in dives if d in input_file][0]

    lens_calibration_path = Path("./data/fsl-01d-lens-raw.pkg")
    laser_calibration_path = Path(laser_params[dive])
    laser_calibration_path = Path("./data/laser-calibration.pkg")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    laser_detector = LaserDetector(Path("./data/models/laser_detection.pth"), lens_calibration_path, laser_calibration_path, device)
    fish_segmentation_inference = FishSegmentationInference(device)

    raw_processor_hist_eq = RawProcessor()
    raw_processor = RawProcessor(enable_histogram_equalization=False)

    image_rectifier = ImageRectifier(lens_calibration_path)

    img = raw_processor_hist_eq.load_and_process(Path(input_file))
    img_dark = raw_processor.load_and_process(Path(input_file))

    img = image_rectifier.rectify(img)
    img_dark = image_rectifier.rectify(img_dark)

    img8 = uint16_2_uint8(img)
    img_dark8 = uint16_2_uint8(img_dark)

    laser_coords = laser_detector.find_laser(img_dark8)

    if laser_coords is None:
        target_file = input_file.replace(".ORF", "_laser_reject.PNG")
        cv2.imwrite(target_file, img8)

        return None, "laser_reject"

    laser_img = cv2.circle(img8.copy(), laser_coords, 8, (0, 0, 255), -1)

    segmentations = fish_segmentation_inference.inference(img8)

    if segmentations.sum() == 0:
        target_file = input_file.replace(".ORF", "_fish_reject.PNG")
        cv2.imwrite(target_file, laser_img)

        return None, "fish_reject"
    
    mask = np.zeros_like(segmentations, dtype=bool)
    mask[segmentations == segmentations[laser_coords[1], laser_coords[0]]] = True

    if mask.sum() == 0:
        target_file = input_file.replace(".ORF", "_fishlaser_reject.PNG")
        cv2.imwrite(target_file, img8)
        
        return None, "fishlaser_reject"

    fish_head_tail_detector = FishHeadTailDetector()
    left_coord, right_coord = fish_head_tail_detector.find_head_tail(mask)

    fish_img = cv2.circle(laser_img, left_coord, 8, (0, 255, 255), -1)
    fish_img = cv2.circle(fish_img, right_coord, 8, (0, 255, 0), -1)

    world_point_handler = WorldPointHandler(lens_calibration_path, laser_calibration_path)
    laser_coords3d = world_point_handler.calculate_laser_parallax(laser_coords)

    left_coord3d, right_coord3d = world_point_handler.calculate_world_coordinates_with_depth(left_coord, right_coord, laser_coords3d[2])

    length = np.linalg.norm(left_coord3d - right_coord3d)

    target_file = input_file.replace(".ORF", "_success.PNG")
    cv2.imwrite(target_file, fish_img)

    return input_file, laser_coords[0], laser_coords[1], left_coord[0], left_coord[1], right_coord[0], right_coord[1], laser_coords3d[2], length


def main():
    files = glob("/home/chris/8-03-23 FSL-01D Florida/**/*.ORF")
    results = [process(f) for f in tqdm(files)]

    results_df = pd.DataFrame(
        np.array([r for r in results if r[0] is not None]),
        columns=["input_file", "laser_coords_0", "laser_coords_1", "left_coord_0", "left_coord_1", "right_coord_0", "right_coord_1", "distance", "length"])
    
    results_df.to_csv("/home/chris/8-03-23 FSL-01D Florida/results.csv")

    status = [r[1] if r[0] is None else "success" for r in results]

    status_df = pd.DataFrame(
        np.array(list(zip(files, status))),
        columns=["input_file", "status"]
    )
    status_df.to_csv("/home/chris/8-03-23 FSL-01D Florida/status.csv")


if __name__ == "__main__":
    main()