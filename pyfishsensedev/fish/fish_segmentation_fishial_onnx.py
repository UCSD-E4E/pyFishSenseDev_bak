from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime

from pyfishsensedev.fish.fish_segmentation_fishial import FishSegmentationFishial


# Adapted from https://github.com/fishial/fish-identification/blob/main/module/segmentation_package/interpreter_segm.py
class FishSegmentationFishialOnnx(FishSegmentationFishial):
    MODEL_URL = "https://huggingface.co/ccrutchf/fishial/resolve/main/fishial.onnx?download=true"
    MODEL_PATH = FishSegmentationFishial._get_model_directory() / "models" / "fishial.onnx"

    def __init__(self) -> None:
        super().__init__()

        self.model_path = self._download_file(
            FishSegmentationFishialOnnx.MODEL_URL,
            FishSegmentationFishialOnnx.MODEL_PATH,
        ).as_posix()

        self.ort_session = onnxruntime.InferenceSession(self.model_path)

    def unwarp_tensor(self, tensor: Tuple) -> Tuple:
        return tensor

    def inference(self, img: np.ndarray) -> np.ndarray:
        resized_img, scales = self._resize_img(img)

        ort_inputs = {
            self.ort_session.get_inputs()[0]
            .name: resized_img.astype("float32")
            .transpose(2, 0, 1)
        }
        ort_outs = self.ort_session.run(None, ort_inputs)

        complete_mask = self._convert_output_to_mask_and_polygons(
            ort_outs, resized_img, scales, img
        )

        return complete_mask[:, :, 0]
