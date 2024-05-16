import os

import numpy as np
import onnx
import onnxoptimizer
import torch.onnx

from pyfishsensedev.fish import FishSegmentationFishialPyTorch


def main():
    os.makedirs("./build", exist_ok=True)
    onnx_file = "./build/fishial.onnx"

    data = np.load("./tests/data/fish_segmentation.npz")
    img = data["img8"]

    fish_segmentation = FishSegmentationFishialPyTorch("cpu")
    fish_segmentation.model.eval()

    resized_img, _ = fish_segmentation._resize_img(img)
    tensor_img = torch.Tensor(resized_img.astype("float32").transpose(2, 0, 1))

    torch.onnx.export(fish_segmentation.model, tensor_img, onnx_file)

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)

    optimized_model = onnxoptimizer.optimize(onnx_model)
    onnx.checker.check_model(optimized_model)

    onnx.save(optimized_model, onnx_file)


if __name__ == "__main__":
    main()
