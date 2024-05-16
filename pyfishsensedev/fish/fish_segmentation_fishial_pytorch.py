from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch
import torchvision  # Needed to load the *.ts torchscript model.
from PIL import Image
from requests import get
from torch.nn import functional as F

from pyfishsensedev.fish.fish_segmentation_fishial import FishSegmentationFishial


# Adapted from https://github.com/fishial/fish-identification/blob/main/module/segmentation_package/interpreter_segm.py
class FishSegmentationFishialPyTorch(FishSegmentationFishial):
    MODEL_URL = (
        "https://storage.googleapis.com/fishial-ml-resources/segmentation_21_08_2023.ts"
    )
    MODEL_PATH = Path("./data/models/fishial.ts")

    def __init__(self, device: str):
        super().__init__()
        self.device = device

        self.model_path = self.__download_file(
            FishSegmentationFishialPyTorch.MODEL_URL,
            FishSegmentationFishialPyTorch.MODEL_PATH,
        ).as_posix()
        self.model = torch.jit.load(self.model_path).to(device).eval()

    def __download_file(self, url: str, path: Path) -> Path:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

            response = get(url)
            with path.open("wb") as file:
                file.write(response.content)

        return path.absolute()

    def unwarp_tensor(self, tensor: Iterable[torch.Tensor]) -> Tuple:
        return (t.cpu().numpy() for t in tensor)

    def inference(self, img: np.ndarray) -> np.ndarray:
        resized_img, scales = self._resize_img(img)

        tensor_img = torch.Tensor(resized_img.astype("float32").transpose(2, 0, 1)).to(
            self.device
        )

        segm_output = self.model(tensor_img)
        complete_mask = self._convert_output_to_mask_and_polygons(
            segm_output, resized_img, scales, img
        )

        return complete_mask[:, :, 0]
