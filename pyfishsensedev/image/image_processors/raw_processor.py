from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import rawpy

from pyfishsensedev.image.image_processors.image_processor import ImageProcessor


class RawProcessor(ImageProcessor):
    def __init__(self, enable_histogram_equalization=True):
        self.enable_histogram_equalization = enable_histogram_equalization

        super().__init__()

    def load_and_process(self, path: Path) -> np.ndarray:
        with rawpy.imread(path.as_posix()) as raw:
            return self.process(raw.raw_image.copy())

    def process(self, raw: np.ndarray) -> np.ndarray:
        # Remove excess data on the edge of the image.
        img = raw[:, :-53]
        img = self._linearization(img)
        img = self._demosaic(img)
        img = self._grey_world_wb(img)
        if self.enable_histogram_equalization:
            img, _, _ = self._histogram_equalize(img)

        return img

    def _demosaic(self, img: np.ndarray) -> np.ndarray:
        return cv2.demosaicing(img, cv2.COLOR_BayerGB2BGR)

    def _linearization(self, img: np.ndarray) -> np.ndarray:
        img[img > 65000] = img.min()
        img = ((img - img.min()) * (1 / (img.max() - img.min()) * 65535)).astype(
            np.uint16
        )

        return img

    def _grey_world_wb(self, img: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(img)
        r_avg = cv2.mean(r)[0]
        g_avg = cv2.mean(g)[0]
        b_avg = cv2.mean(b)[0]

        kr = g_avg / r_avg
        kg = 1
        kb = g_avg / b_avg

        r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

        return cv2.merge([b, g, r])

    def _hist_eq(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function will do histogram equalization on a given 1D np.array
        meaning will balance the colors in the image.
        For more details:
        https://en.wikipedia.org/wiki/Histogram_equalization
        **Original function was taken from open.cv**
        :param img: a 1D np.array that represent the image
        :return: imgnew -> image after equalization, hist-> original histogram, histnew -> new histogram
        """

        # Flattning the image and converting it into a histogram
        histOrig, bins = np.histogram(img.flatten(), 65536, [0, 65535])
        # Calculating the cumsum of the histogram
        cdf = histOrig.cumsum()
        # Places where cdf = 0 is ignored and the rest is stored
        # in cdf_m
        cdf_m = np.ma.masked_equal(cdf, 0)
        # Normalizing the cdf
        cdf_m = (cdf_m - cdf_m.min()) * 65535 / (cdf_m.max() - cdf_m.min())
        # Filling it back with zeros
        cdf = np.ma.filled(cdf_m, 0)

        # Creating the new image based on the new cdf
        imgEq = cdf[img.astype("uint16")]
        histEq, _ = np.histogram(imgEq.flatten(), 65536, [0, 65536])

        return imgEq, histOrig, histEq

    def _histogram_equalize(
        self, imgOrig: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Equalizes the histogram of an image
        The function will fist check if the image is RGB or gray scale
        If the image is gray scale will equalizes
        If RGB will first convert to YIQ then equalizes the Y level
        :param imgOrig: Original Histogram
        :return: imgnew -> image after equalization, hist-> original histogram, histnew -> new histogram
        """

        if len(imgOrig.shape) == 2:
            img = imgOrig * 65536
            imgEq, histOrig, histEq = self._hist_eq(img)

        else:
            img = imgOrig
            img[:, :, 0], histOrig, histEq = self._hist_eq(img[:, :, 0])
            img[:, :, 1], histOrig, histEq = self._hist_eq(img[:, :, 1])
            img[:, :, 2], histOrig, histEq = self._hist_eq(img[:, :, 2])
            imgEq = img

        return imgEq, histOrig, histEq


if __name__ == "__main__":
    import os
    from glob import glob
    from multiprocessing import Pool, cpu_count

    from tqdm import tqdm

    def process(raw: str):
        new_path = raw.replace("raw", "png").replace(".ORF", ".png")
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        if os.path.exists(new_path):
            return

        raw_processor = RawProcessor()

        cv2.imwrite(new_path, raw_processor.load_and_process(Path(raw)))

    raw_images = glob("./data/raw/**/*.ORF", recursive=True)
    raw_images.sort()
    list(tqdm(Pool(cpu_count()).imap(process, raw_images), total=len(raw_images)))
