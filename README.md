# Ant / Bee Pipeline

```
Image -> YOLO26n detector -> crop each insect -> YOLO11s classifier -> decision engine -> Ant / Bee / Other / Unknown
```

## Setup
```bash
pip install -r requirements.txt
python -m src.inspect_models     # sanity check: prints classes of both weights
```

## Web app
```bash
python app.py                    # http://localhost:5000
```
Upload single/multiple images or a `.zip` (up to 100 images per upload; change `MAX_IMAGES` in `app.py`).
Annotated images are also saved to `outputs/`.

## CLI
```bash
python -m src.predict test_images/            # folder, single image or .zip
```

## Decision engine (`src/pipeline.py::decide`)
Detector classes (Ant, Bee, Beetle, Bug, Fly, Lep, Spider, Wasp) and classifier classes
(ants, bees, butterflies, flies, moths, spiders, wasps) are mapped to ant / bee / other.
- Both agree and confidence >= 0.35 -> that label (agree on "other" -> **Other**)
- Disagree -> the more confident model wins if it clears its threshold (classifier 0.60, detector 0.40)
- Otherwise -> **Unknown**

Thresholds live in `DecisionConfig`.
