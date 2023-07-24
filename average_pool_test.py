import os
from pathlib import Path
import json
import cv2
import numpy as np

from camera_imaging_pipeline.src.image_processing import imageProcessing
from detect_laser import get_masked_image_matrix

def mean_pooling(kernel_shape, img):

    M, N = np.shape(img)
    K = kernel_shape[0]
    L = kernel_shape[1]

    MK = M // K
    NL = N // L

    res = img[:MK*K, :NL*L].reshape(MK, K, NL, L).mean(axis=(1, 3))

    return res

laser_path = Path("C:/Users/Hamish/Documents/E4E\Fishsense/fishsense-lite-python-pipeline/files/laser-calibration-output-7-13.dat")
calibration_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/files/fsl-01d-lens.dat")
data_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/7_23_nathans_pool/FSL-01F_Fred")

params_path = Path('C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/camera_imaging_pipeline/params1.json')
with open(params_path, 'r', encoding='ascii') as handle:
    params = json.load(handle)

files = list(data_path.glob('*.ORF'))

processor = imageProcessing()

for filepath in sorted(files):
    print(f"Processing {filepath.stem}")

    processed_image, _ = processor.applyToImage(filepath.as_posix(), params)
    
    print(len(np.shape(processed_image)))
    masked_image = get_masked_image_matrix(laser_path, calibration_path, processed_image)

    # pooled = mean_pooling((2,2), masked_image).astype(np.uint8)
    # pooled = mean_pooling((2,2), pooled).astype(np.uint8)

    # max_index = np.unravel_index(pooled.argmax(), pooled.shape)

    # #pooled[max_index[0]][max_index[1]] = 0
    # cv2.circle(pooled, (max_index[0],max_index[1]), 100, 255, cv2.FILLED, cv2.LINE_AA)

    cv2.namedWindow("Zoomed", cv2.WINDOW_NORMAL)
    cv2.imshow("Zoomed", processed_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# processed_image, _ = processor.applyToImage(filepath.as_posix(), params)

# # print(len(np.shape(processed_image)))
# # masked_image = get_masked_image_matrix(laser_path, calibration_path, processed_image)

# # pooled = mean_pooling((2,2), masked_image).astype(np.uint8)
# # pooled = mean_pooling((2,2), pooled).astype(np.uint8)

# # max_index = np.unravel_index(pooled.argmax(), pooled.shape)

# # pooled[max_index[0]][max_index[1]] = 255

# cv2.namedWindow("Zoomed", cv2.WINDOW_NORMAL)
# cv2.imshow("Zoomed", processed_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()