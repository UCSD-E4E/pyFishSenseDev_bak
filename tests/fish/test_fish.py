import numpy as np
import torch

from pyfishsensedev.fish.fish_segmentation_pytorch import FishSegmentationPyTorch


def test_fishial_pytorch():
    data = np.load("./tests/data/fish_segmentation.npz")
    img8 = data["img8"]
    truth = data["segmentations"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fish_segmentation = FishSegmentationPyTorch(device)
    result = fish_segmentation.inference(img8)

    np.testing.assert_array_equal(result, truth)
