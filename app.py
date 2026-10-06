"""Web app: python app.py  ->  http://localhost:5000"""
import base64, io, zipfile
from pathlib import Path
import cv2, numpy as np
from flask import Flask, jsonify, request, send_from_directory
from src.pipeline import Pipeline

MAX_IMAGES = 100                      # per request (zip or multi-file); >= 50 required
MAX_UPLOAD_MB = 300
MAX_UNZIPPED_MB = 500                 # zip-bomb guard
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT = Path(__file__).parent / "outputs"; OUT.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
_pipe = None


def pipe():
    global _pipe
    if _pipe is None:
        _pipe = Pipeline()
    return _pipe


def collect(files):
    """Yield (name, bytes) from uploaded images and/or zips."""
    items = []
    for f in files:
        name = Path(f.filename or "image").name
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(f.stream) as z:
                infos = [i for i in z.infolist() if not i.is_dir()
                         and Path(i.filename).suffix.lower() in EXT
                         and not Path(i.filename).name.startswith(("._", "."))
                         and "__MACOSX" not in i.filename]
                if sum(i.file_size for i in infos) > MAX_UNZIPPED_MB * 1024 * 1024:
                    raise ValueError("Zip is too large when extracted.")
                items += [(Path(i.filename).name, z.read(i)) for i in infos]
        elif Path(name).suffix.lower() in EXT:
            items.append((name, f.read()))
    return items


def b64_jpg(img, max_side=1280, q=85):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/config")
def config():
    return jsonify(max_images=MAX_IMAGES, max_upload_mb=MAX_UPLOAD_MB)


@app.post("/api/predict")
def predict():
    try:
        items = collect(request.files.getlist("files"))
    except (zipfile.BadZipFile, ValueError) as e:
        return jsonify(error=str(e) or "Invalid zip file."), 400
    if not items:
        return jsonify(error="No valid images found (jpg, png, bmp, webp or a zip of them)."), 400
    if len(items) > MAX_IMAGES:
        return jsonify(error=f"Too many images ({len(items)}). Limit is {MAX_IMAGES} per upload."), 400

    p, results = pipe(), []
    for name, data in items:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            results.append(dict(name=name, error="Could not decode image"))
            continue
        r = p.run(img)
        vis = p.annotate(img, r)
        cv2.imwrite(str(OUT / f"{Path(name).stem}_pred.jpg"), vis)
        results.append(dict(
            name=name, counts=r.counts, discarded=r.discarded, image=b64_jpg(vis),
            insects=[dict(box=i.box, final=i.final, reason=i.reason,
                          detector=dict(label=i.detector_label, conf=round(i.detector_conf, 3)),
                          classifier=dict(label=i.classifier_label, conf=round(i.classifier_conf, 3)))
                     for i in r.insects]))
    return jsonify(results=results)


if __name__ == "__main__":
    pipe()  # warm up so first request is fast
    app.run(host="0.0.0.0", port=5000, debug=False)
