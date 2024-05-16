from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime

from pyfishsensedev.fish.fish_segmentation_fishial import FishSegmentationFishial


# Adapted from https://github.com/fishial/fish-identification/blob/main/module/segmentation_package/interpreter_segm.py
class FishSegmentationFishialOnnx(FishSegmentationFishial):
    def __init__(self) -> None:
        super().__init__()

        parent = Path(__file__).parent
        model_path = parent / "models" / "fishial.onnx"

        self.ort_session = onnxruntime.InferenceSession(
            model_path.absolute().as_posix()
        )

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
