import os

import numpy as np
import onnx
import torch.onnx

from pyfishsensedev.fish import FishSegmentationFishialPyTorch


def main():
    os.makedirs("./build", exist_ok=True)

    data = np.load("./tests/data/fish_segmentation.npz")
    img = data["img8"]

    fish_segmentation = FishSegmentationFishialPyTorch("cpu")
    fish_segmentation.model.eval()

    resized_img = fish_segmentation._resize_img(img)
    tensor_img = torch.Tensor(resized_img.astype("float32").transpose(2, 0, 1))

    torch.onnx.export(fish_segmentation.model, tensor_img, "./build/fishial.onnx")

    onnx_model = onnx.load("./build/fishial.onnx")
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    main()
