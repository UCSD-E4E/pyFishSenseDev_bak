import numpy as np
import pytest
import torch

from pyfishsensedev.fish import (
    FishSegmentationFishialOnnx,
    FishSegmentationFishialPyTorch,
)
from pyfishsensedev.fish.fish_segmentation import FishSegmentation


@pytest.mark.parametrize(
    "fish_segmentation",
    [
        FishSegmentationFishialPyTorch("cuda" if torch.cuda.is_available() else "cpu"),
        FishSegmentationFishialOnnx(),
    ],
)
def test_fishial(fish_segmentation: FishSegmentation):
    data = np.load("./tests/data/fish_segmentation.npz")
    img8 = data["img8"]
    truth = data["segmentations"]

    result = fish_segmentation.inference(img8)
    np.testing.assert_array_equal(result, truth)
