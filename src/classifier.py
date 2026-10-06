"""Stage 2: YOLO11s classifier - classifies a single cropped insect."""
from pathlib import Path
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models/classifier/yolo11s_cls_low_lr_best.pt"


class InsectClassifier:
    def __init__(self, weights=DEFAULT_WEIGHTS, imgsz=320, device=None):
        self.model = YOLO(str(weights))
        self.imgsz, self.device = imgsz, device

    def classify(self, crop: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        """Returns (top1 label, top1 conf, top3 [(label, conf), ...])."""
        res = self.model.predict(crop, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        p = res.probs
        top3 = [(res.names[int(i)], float(p.data[int(i)])) for i in p.top5[:3]]
        return res.names[int(p.top1)], float(p.top1conf), top3
