"""Find out WHICH model is failing, by running each one on its own.

    python -m src.diagnose test_images/            (folder, image or .zip)
    python -m src.diagnose test_images/ --det-conf 0.10

Per image it prints:
  [DET]  every detector box (label + conf) at a LOW threshold, so you see what it finds / misses
  [CLS]  classifier top-3 for each crop, and for the whole image as a sanity check
Saves to outputs/diagnose/:  <img>_det.jpg (detector-only boxes) and crops/ (what the classifier saw)
"""
import argparse
from pathlib import Path
import cv2
from .detector import InsectDetector
from .classifier import InsectClassifier
from .pipeline import norm
from .predict import iter_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--det-conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640, help="detector inference size (try 960/1280 for crowds)")
    ap.add_argument("--pad", type=float, default=0.10)
    a = ap.parse_args()

    out = Path("outputs/diagnose"); (out / "crops").mkdir(parents=True, exist_ok=True)
    det, cls = InsectDetector(conf=a.det_conf, imgsz=a.imgsz), InsectClassifier()
    n_img = n_box = n_agree = n_empty = 0

    for name, img in iter_inputs(Path(a.input)):
        if img is None:
            continue
        n_img += 1
        stem, (h, w) = Path(name).stem, img.shape[:2]
        dets = det.detect(img)
        vis = img.copy()
        print(f"\n=== {name}  ({w}x{h})  detector boxes: {len(dets)}")
        if not dets:
            n_empty += 1
        for k, d in enumerate(dets):
            x1, y1, x2, y2 = d.box
            px, py = int((x2 - x1) * a.pad), int((y2 - y1) * a.pad)
            crop = img[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
            if crop.size == 0:
                continue
            label, conf, top3 = cls.classify(crop)
            agree = norm(d.label) == norm(label)
            n_box += 1; n_agree += agree
            area = (x2 - x1) * (y2 - y1) / (w * h)
            print(f"  [DET] #{k} {d.label:<7} {d.conf:.2f}  box covers {area:.0%} of image")
            print(f"  [CLS]    {', '.join(f'{l} {c:.2f}' for l, c in top3)}   -> {'AGREE' if agree else 'DISAGREE'}")
            cv2.imwrite(str(out / "crops" / f"{stem}_{k}_det-{d.label}_cls-{label}.jpg"), crop)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 3)
            cv2.putText(vis, f"{d.label} {d.conf:.2f}", (x1 + 3, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
        full = cls.classify(img)
        print(f"  [CLS whole image] {', '.join(f'{l} {c:.2f}' for l, c in full[2])}")
        cv2.imwrite(str(out / f"{stem}_det.jpg"), vis)

    print(f"\n--- SUMMARY: {n_img} images | {n_box} detector boxes | {n_empty} images with NO boxes"
          f" | detector/classifier agree on {n_agree}/{n_box}")


if __name__ == "__main__":
    main()
