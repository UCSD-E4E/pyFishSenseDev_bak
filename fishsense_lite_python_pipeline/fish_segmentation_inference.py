from pathlib import Path
from zipfile import ZipFile

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from projects.PointRend.point_rend import add_pointrend_config
from requests import get
import cv2
import numpy as np
import torch

class FishSegmentationInference:
    MODEL_URL = "https://storage.googleapis.com/fishial-ml-resources/model_15_11_2022.pth"
    MODEL_PATH = Path("./data/models/model_15_11_2022.pth")
    DECTECTRON2_ZIP_URL = "https://github.com/facebookresearch/detectron2/archive/refs/heads/main.zip"
    DECTECTRON2_ZIP_PATH = Path("./data/detectron2.zip")
    CONFIG_PATH = Path("./data/detectron2/detectron2-main/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")

    def __init__(self, device: str):
        self.model_path = self._download_file(
            FishSegmentationInference.MODEL_URL,
            FishSegmentationInference.MODEL_PATH).as_posix()

        self.cfg = get_cfg()
        add_pointrend_config(self.cfg)
        self.cfg.merge_from_file(self._get_config().as_posix())
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.WEIGHTS = self.model_path
        self.cfg.freeze()
        self.model = DefaultPredictor(self.cfg)

    def _download_file(self, url: str, path: Path) -> Path:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

            response = get(url)
            with path.open("wb") as file:
                file.write(response.content)

        return path.absolute()
    
    def _get_config(self) -> Path:
        detectron2_zip_path = self._download_file(
            FishSegmentationInference.DECTECTRON2_ZIP_URL,
            FishSegmentationInference.DECTECTRON2_ZIP_PATH)
        
        detectron2_path = Path("./data/detectron2")
        if not detectron2_path.exists():
            with ZipFile(detectron2_zip_path) as zip:
                zip.extractall(detectron2_path)

        return FishSegmentationInference.CONFIG_PATH.absolute()

    def inference(self, img: np.ndarray) -> np.ndarray:
        outputs = self.model(img)

        complete_mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
        for idx, mask in enumerate(outputs['instances'].pred_masks):
            complete_mask += (idx + 1) * mask.cpu()
        
        return complete_mask
    
if __name__ == "__main__":
    img = cv2.imread("./data/png/P7170081.png")

    fish_segmentation_inference = FishSegmentationInference('cuda' if torch.cuda.is_available() else 'cpu')
    outputs = fish_segmentation_inference.inference(img)

    print(outputs)