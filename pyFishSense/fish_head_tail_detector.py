from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

class FishHeadTailDetector:
    def find_head_tail(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        y, x = mask.nonzero()
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        mask_crop = mask[y_min:y_max, x_min:x_max]
        y, x = mask_crop.nonzero()

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        x = x - x_mean
        y = y - y_mean
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)

        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue

        height, width = mask_crop.shape

        scale = height if height > width else width

        coord1 = np.array([
            -x_v1*scale*2,
            -y_v1*scale*2
        ])

        coord2 = np.array([
            x_v1*scale*2,
            y_v1*scale*2
        ])

        extent = [x.min(), x.max(), y.min(), y.max()] 

        # min_x = coord1[0] if coord1[0] < coord2[0] else coord2[0]
        # min_y = coord1[1] if coord1[1] < coord2[1] else coord2[1]

        # coord1[0] += np.mean(x)
        # coord2[0] += np.mean(x)

        # coord1[1] += np.mean(y)
        # coord1[1] += np.mean(y)

        # coord1[0] -= min_x
        # coord2[0] -= min_x

        # coord1[1] -= min_y
        # coord2[1] -= min_y

        coord1[0] -= x_min
        coord2[0] += x_min

        coord1[1] -= y_min
        coord2[1] += y_min

        # coord1[0] -= x_mean
        # coord2[0] -= x_mean

        # coord1[1] -= y_mean
        # coord2[1] -= y_mean

        m = y_v1 / x_v1
        b = coord1[1] - m * coord1[0]

        y, x = mask_crop.nonzero()
        y_target = m * x + b

        x_pts = [0, width]
        y_pts = np.array([b, m * width + b])

        points_along_line = np.where(np.abs(y - y_target) < 1)
        x = x[points_along_line]
        y = y[points_along_line]
        
        coords = np.stack([x, y])
        left_coord = coords[:, np.argmin(x)]
        right_coord = coords[:, np.argmax(x)]

        y, x = mask.nonzero()
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        left_coord[0] += x_min
        right_coord[0] += x_min
        left_coord[1] += y_min
        right_coord[1] += y_min

        return left_coord, right_coord

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    import torch

    from pyFishSense.fish_segmentation_inference import FishSegmentationInference
    from pyFishSense.laser_detector import LaserDetector
    from pyFishSense.image_rectifier import ImageRectifier
    from pyFishSense.raw_processor import RawProcessor

    raw_processor = RawProcessor()
    raw_processor_dark = RawProcessor(enable_histogram_equalization=False)
    image_rectifier = ImageRectifier(Path("./data/lens-calibration.pkg"))
    laser_detector = LaserDetector(
        Path("./data/models/laser_detection.pth"),
        Path("./data/lens-calibration.pkg"),
        Path("./data/laser-calibration.pkg"))

    img = raw_processor.load_and_process(Path("./data/P8030201.ORF"))
    img_dark = raw_processor_dark.load_and_process(Path("./data/P8030201.ORF"))
    img = image_rectifier.rectify(img)
    img_dark = image_rectifier.rectify(img_dark)

    img8 = ((img.astype('float64') / 65535) * 255).astype('uint8')
    img_dark8 = ((img_dark.astype('float64') / 65535) * 255).astype('uint8')
    coords = laser_detector.find_laser(img_dark8)

    fish_segmentation_inference = FishSegmentationInference('cuda' if torch.cuda.is_available() else 'cpu')
    segmentations = fish_segmentation_inference.inference(img8)

    mask = np.zeros_like(segmentations, dtype=bool)
    mask[segmentations == segmentations[coords[1], coords[0]]] = True

    fish_head_tail_detector = FishHeadTailDetector()
    left_coord, right_coord = fish_head_tail_detector.find_head_tail(mask, img8)

    plt.imshow(img8)
    plt.plot(left_coord[0], left_coord[1], 'r.')
    plt.plot(right_coord[0], right_coord[1], 'b.')
    plt.show()