"""Print class names / task / image size of both weights: python -m src.inspect_models"""
from ultralytics import YOLO
from .detector import DEFAULT_WEIGHTS as D
from .classifier import DEFAULT_WEIGHTS as C

for p in (D, C):
    m = YOLO(str(p))
    print(p.name, "| task:", m.task, "| imgsz:", m.overrides.get("imgsz"))
    print("  classes:", m.names)
