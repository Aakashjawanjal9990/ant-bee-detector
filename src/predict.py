"""CLI: python -m src.predict test_images/            (folder, file, or .zip)"""
import argparse, json, zipfile
from pathlib import Path
import cv2, numpy as np
from .pipeline import Pipeline

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_inputs(path: Path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in sorted(z.namelist()):
                if Path(n).suffix.lower() in EXT and not Path(n).name.startswith("."):
                    yield Path(n).name, cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
    else:
        files = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.suffix.lower() in EXT)
        for f in files:
            yield f.name, cv2.imread(str(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--det-conf", type=float, default=0.25)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    pipe, report = Pipeline(det_conf=a.det_conf), {}
    for name, img in iter_inputs(Path(a.input)):
        if img is None:
            continue
        r = pipe.run(img)
        cv2.imwrite(str(out / f"{Path(name).stem}_pred.jpg"), pipe.annotate(img, r))
        report[name] = r.counts
        print(name, r.counts)
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
