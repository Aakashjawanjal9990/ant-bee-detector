"""Stage 1: YOLO26n detector - finds individual insects in an image."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models/detector/best.pt"


@dataclass
class Detection:
    box: tuple          # (x1, y1, x2, y2) ints
    label: str          # detector class name
    conf: float


class InsectDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.25, imgsz=640, device=None):
        self.model = YOLO(str(weights))
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def detect(self, image: np.ndarray) -> list[Detection]:
        res = self.model.predict(image, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        names, out = res.names, []
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append(Detection((x1, y1, x2, y2), names[int(b.cls)], float(b.conf)))
        return out
