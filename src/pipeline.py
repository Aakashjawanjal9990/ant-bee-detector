"""Detector -> crop -> classifier -> decision engine."""
from dataclasses import dataclass, field
import cv2
import numpy as np
from .detector import InsectDetector
from .classifier import InsectClassifier

# Normalise both models' vocabularies to a common key
_NORM = {"ant": "ant", "ants": "ant", "bee": "bee", "bees": "bee"}
# If you retrain the classifier with a "not an insect" class, any of these names is treated as a reject
_REJECT = {"background", "not_insect", "non_insect", "noninsect", "negative", "none", "nothing", "reject"}
FINAL_COLORS = {"Ant": (60, 60, 220), "Bee": (0, 200, 255), "Other": (200, 120, 40), "Unknown": (150, 150, 150)}


def norm(label: str) -> str:
    l = label.lower().replace(" ", "_")
    return "reject" if l in _REJECT else _NORM.get(l, "other")


@dataclass
class DecisionConfig:
    min_cls_conf: float = 0.60       # classifier confidence needed to override the detector
    min_det_conf: float = 0.40       # detector confidence needed to override the classifier
    agree_conf: float = 0.35         # lower bar when both models agree
    margin: float = 0.05             # winner must beat the other model by this much, else Unknown
    strong_det: float = 0.50         # detector conf below this = "weak" detection
    weak_confirm_conf: float = 0.80  # weak boxes survive only if the classifier is this sure
    drop_weak_unconfirmed: bool = True   # discard (not label) weak boxes the classifier can't confirm
    reject_conf: float = 0.50        # classifier 'background' class at/above this -> box discarded


def dedupe(dets, iou_thr=0.5, contain_thr=0.9):
    """Class-agnostic duplicate removal (YOLO26 is NMS-free, so overlaps survive)."""
    def inter(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
        return max(0, w) * max(0, h)
    def area(b):
        return max(1, (b[2] - b[0]) * (b[3] - b[1]))
    keep = []
    for d in sorted(dets, key=lambda d: -d.conf):
        dup = False
        for k in keep:
            i = inter(d.box, k.box)
            if i / (area(d.box) + area(k.box) - i) >= iou_thr or i / area(d.box) >= contain_thr:
                dup = True; break
        if not dup:
            keep.append(d)
    return keep


def decide(det_label, det_conf, cls_label, cls_conf, cfg: DecisionConfig):
    """Decision engine -> (Ant|Bee|Other|Unknown | None to discard, reason)."""
    d, c = norm(det_label), norm(cls_label)
    if c == "reject":                                   # classifier trained with a background class
        return (None, "classifier: not an insect") if cls_conf >= cfg.reject_conf else ("Unknown", "possible non-insect")
    if det_conf < cfg.strong_det:                       # weak detection: needs classifier confirmation
        if cls_conf >= cfg.weak_confirm_conf and (c in ("ant", "bee") or c == d):
            return c.capitalize(), "weak detection confirmed by classifier"
        if cfg.drop_weak_unconfirmed:
            return None, "weak detection not confirmed"
        return "Unknown", "weak detection not confirmed"
    if d == c:
        if min(det_conf, cls_conf) >= cfg.agree_conf:
            return c.capitalize(), "detector and classifier agree"
        return "Unknown", "models agree but confidence is low"
    if cls_conf >= cfg.min_cls_conf and cls_conf >= det_conf + cfg.margin:
        return c.capitalize(), "classifier overrode detector"
    if det_conf >= cfg.min_det_conf and det_conf >= cls_conf + cfg.margin:
        return d.capitalize(), "detector overrode classifier"
    return "Unknown", "models disagree"


@dataclass
class InsectResult:
    box: tuple
    detector_label: str
    detector_conf: float
    classifier_label: str
    classifier_conf: float
    top3: list
    final: str
    reason: str


@dataclass
class ImageResult:
    insects: list = field(default_factory=list)
    discarded: int = 0          # weak, unconfirmed boxes that were ignored

    @property
    def counts(self):
        c = {"Ant": 0, "Bee": 0, "Other": 0, "Unknown": 0}
        for i in self.insects:
            c[i.final] += 1
        return c


class Pipeline:
    def __init__(self, det_conf=0.25, pad=0.10, cfg: DecisionConfig | None = None, device=None):
        self.detector = InsectDetector(conf=det_conf, device=device)
        self.classifier = InsectClassifier(device=device)
        self.pad, self.cfg = pad, cfg or DecisionConfig()

    def _crop(self, img, box):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * self.pad), int((y2 - y1) * self.pad)
        return img[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]

    def run(self, img: np.ndarray) -> ImageResult:
        out = ImageResult()
        for d in dedupe(self.detector.detect(img)):
            crop = self._crop(img, d.box)
            if crop.size == 0:
                continue
            c_label, c_conf, top3 = self.classifier.classify(crop)
            final, why = decide(d.label, d.conf, c_label, c_conf, self.cfg)
            if final is None:
                out.discarded += 1
                continue
            out.insects.append(InsectResult(d.box, d.label, d.conf, c_label, c_conf, top3, final, why))
        return out

    @staticmethod
    def annotate(img: np.ndarray, result: ImageResult) -> np.ndarray:
        vis = img.copy()
        th = max(1, round(min(vis.shape[:2]) / 300))
        for i in result.insects:
            x1, y1, x2, y2 = i.box
            col = FINAL_COLORS[i.final]
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, th * 2)
            txt = f"{i.final} {min(i.detector_conf, i.classifier_conf):.0%}"
            fs = th * 0.6
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            cv2.rectangle(vis, (x1, max(0, y1 - tht - 8)), (x1 + tw + 6, y1), col, -1)
            cv2.putText(vis, txt, (x1 + 3, max(tht, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
        return vis
