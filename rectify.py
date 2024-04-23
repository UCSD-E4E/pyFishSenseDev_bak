from glob import glob
from pathlib import Path
from tqdm import tqdm
import cv2

from pyfishsense import FishHeadTailDetector, FishSegmentationInference, ImageRectifier, LaserDetector, RawProcessor, WorldPointHandler

def process(input_file: str):
    lens_calibration_path = Path("./data/fsl-07d-lens-raw.pkg")
    
    raw_processor = RawProcessor(enable_histogram_equalization=False)
    image_rectifier = ImageRectifier(lens_calibration_path)

    img = raw_processor.load_and_process(Path(input_file))
    img = image_rectifier.rectify(img)

    target_file = input_file.replace(".ORF", ".PNG")
    cv2.imwrite(target_file, img)


def main():
    files = glob("/home/chris/data/*.ORF")
    results = [process(f) for f in tqdm(files)]

if __name__ == "__main__":
    main()