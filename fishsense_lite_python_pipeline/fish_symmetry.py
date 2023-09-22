from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

class FishSymmetry:
    def find_symmetry(self, mask: np.ndarray, img) -> Tuple[int, int, int, int]:
        
        y, x = mask.nonzero()
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        mask_crop = mask[y_min:y_max, x_min:x_max]
        img_crop = img[y_min:y_max, x_min:x_max, :]
        y, x = mask_crop.nonzero()

        x = x - np.mean(x)
        y = y - np.mean(y)

        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)

        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evecs[:, sort_indices[1]]


        scale = 300

        extent = [x.min(), x.max(), y.min(), y.max()]  

        plt.imshow(img_crop, extent=extent, origin='lower')

        plt.plot([x_v1*-scale*2, x_v1*scale*2],
                [y_v1*-scale*2, y_v1*scale*2], color='red')
        plt.axis('equal')
        plt.gca().invert_yaxis()  # Match the image system with origin at top left

        # plt.imshow(crop)
        plt.show()

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    import torch

    from fish_segmentation_inference import FishSegmentationInference

    img = cv2.imread("./data/png/P7170081.png")

    fish_segmentation_inference = FishSegmentationInference('cuda' if torch.cuda.is_available() else 'cpu')
    mask = fish_segmentation_inference.inference(img)

    fish_symmetry = FishSymmetry()
    fish_symmetry.find_symmetry(mask, img)

    # plt.imshow(img)
    # plt.show()