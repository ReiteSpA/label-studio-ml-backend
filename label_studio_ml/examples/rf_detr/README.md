# RF-DETR ML Backend for Label Studio

Object detection using [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow) with support for **your own checkpoint**. This backend follows the [Label Studio ML backend guide](https://labelstud.io/guide/ml_create).

## Features

- Use pretrained COCO weights or **load your own trained checkpoint**
- Configurable model size: `nano`, `small`, `medium`, `large`
- Optional custom class names (e.g. for fine-tuned models)
- Outputs bounding boxes as Label Studio `RectangleLabels` for pre-annotations

## Labeling config

Your project should use **RectangleLabels** and **Image**:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Person" background="green"/>
    <Label value="Car" background="blue"/>
  </RectangleLabels>
</View>
```

Label values must match the model’s class names (COCO or your `CLASSES_FILE`).

### Interactive pre-annotations

The backend supports **interactive pre-annotations**. Enable it in your project (e.g. **Settings → Machine Learning → Interactive pre-annotations**), then in the labeling config add a control that sends keypoints or a region:

- **KeypointLabels** (e.g. user clicks on the image): the backend returns only detections that **contain a positive keypoint** (and excludes boxes containing a **negative** keypoint, e.g. Alt+click). So the labeler can click on an object to “focus” predictions on that area.
- **RectangleLabels** used as a focus region: predictions are filtered to boxes that **overlap** that rectangle.

Example: add a KeypointLabels control bound to the same Image so the user can click to refine which boxes are shown. The `predict()` method receives this as `context` and filters results accordingly. With no context (e.g. first load), the backend returns all detections as usual.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CHECKPOINT_PATH` | Path to your custom `.pt` / `.pth` checkpoint. Omit to use pretrained COCO. | (none) |
| `MODEL_SIZE` | When not using a checkpoint: `nano`, `small`, `medium`, `large` | `medium` |
| `CLASSES_FILE` | Path to class names: JSON array or newline-separated file. Omit for COCO. | (none) |
| `SCORE_THRESHOLD` | Confidence threshold 0–1 | `0.5` |
| `DEVICE` | `cpu` or `cuda` | `cuda` if available |
| `LABEL_STUDIO_URL` | Label Studio URL (for resolving image URLs) | — |
| `LABEL_STUDIO_API_KEY` | Label Studio API key | — |

## Using your own checkpoint

1. Train or export your RF-DETR model (e.g. with [rf-detr training](https://github.com/roboflow/rf-detr)).
2. Place the checkpoint file on the server (e.g. `./data/models/checkpoint.pt`).
3. Set `CHECKPOINT_PATH` to that path (e.g. in `docker-compose.yml` or env).
4. If your model has custom classes, create a `CLASSES_FILE` (e.g. JSON `["class1","class2"]` or one class per line) and set `CLASSES_FILE` to its path.
5. Ensure the Label Studio labeling config uses the same label names as in `CLASSES_FILE`.

Checkpoint format: the backend expects a state dict (e.g. from `torch.save(model.state_dict(), path)` or a dict with a `"state_dict"` or `"model"` key). If your checkpoint layout differs, you may need to adjust loading in `model.py`.

## Run with Docker

```bash
cd label_studio_ml/examples/rf_detr
# Optional: put your checkpoint in ./data/models/checkpoint.pt and set CHECKPOINT_PATH in docker-compose.yml
docker-compose up
```

Backend URL: `http://localhost:9090`. In Label Studio: **Settings → Model** and add this URL.

## Run without Docker

1. Install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend#quickstart) and go to the repo root.

2. Create a virtualenv and install deps:

```bash
cd label_studio_ml/examples/rf_detr
pip install -r requirements-base.txt
pip install -r requirements.txt
```

3. Set env vars (optional: `CHECKPOINT_PATH`, `CLASSES_FILE`, `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_KEY`, etc.).

4. Start the backend (from the `rf_detr` directory or pass the path to it):

```bash
label-studio-ml start . -p 9090
```

Or from the repo root:

```bash
label-studio-ml start label_studio_ml/examples/rf_detr -p 9090
```

5. In Label Studio, connect the model at `http://localhost:9090` (use `http://host.docker.internal:9090` if LS runs in Docker).

## Test

```bash
pip install -r requirements-test.txt
pytest test_api.py -v
```

## References

- [Write your own ML backend](https://labelstud.io/guide/ml_create)
- [RF-DETR](https://github.com/roboflow/rf-detr)
