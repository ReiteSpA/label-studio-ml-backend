"""
RF-DETR ML backend for Label Studio.

Object detection using RF-DETR (Roboflow) with support for custom checkpoints.
See: https://github.com/roboflow/rf-detr
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (
    get_image_size,
    DATA_UNDEFINED_NAME,
)
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import (
    get_data_dir,
    get_local_path,
)

logger = logging.getLogger(__name__)

# Environment configuration
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "")
MODEL_SIZE = os.environ.get("MODEL_SIZE", "medium").lower()
CLASSES_FILE = os.environ.get("CLASSES_FILE", "")
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.5"))
DEVICE = os.environ.get("DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu")
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "false").lower() == "true"

# Lazy-loaded model and class names
_model = None
_class_names = None


def _get_rfdetr_model_class():
    """Return the RF-DETR model class for the given MODEL_SIZE."""
    from rfdetr import (
        RFDETRNano,
        RFDETRSmall,
        RFDETRMedium,
        RFDETRLarge,
    )
    size_map = {
        "nano": RFDETRNano,
        "n": RFDETRNano,
        "small": RFDETRSmall,
        "s": RFDETRSmall,
        "medium": RFDETRMedium,
        "m": RFDETRMedium,
        "large": RFDETRLarge,
        "l": RFDETRLarge,
    }
    cls = size_map.get(MODEL_SIZE, RFDETRMedium)
    return cls


def _load_class_names() -> List[str]:
    """Load class names from CLASSES_FILE or use COCO classes."""
    global _class_names
    if _class_names is not None:
        return _class_names
    if CLASSES_FILE and os.path.isfile(CLASSES_FILE):
        with open(CLASSES_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                _class_names = json.loads(content)
            else:
                _class_names = [line.strip() for line in content.splitlines() if line.strip()]
        logger.info(f"Loaded {len(_class_names)} classes from {CLASSES_FILE}")
    else:
        from rfdetr.util.coco_classes import COCO_CLASSES
        _class_names = COCO_CLASSES
        logger.info(f"Using COCO classes ({len(_class_names)} classes)")
    return _class_names


def _load_model():
    """Load RF-DETR model, optionally from custom checkpoint."""
    global _model
    if _model is not None:
        return _model
    import torch
    model_cls = _get_rfdetr_model_class()
    if CHECKPOINT_PATH and os.path.isfile(CHECKPOINT_PATH):
        logger.info(f"Loading custom checkpoint from {CHECKPOINT_PATH}")
        _model = model_cls(device=DEVICE, pretrain_weights=CHECKPOINT_PATH)
    else:
        logger.info(f"Using pretrained {MODEL_SIZE} model (no CHECKPOINT_PATH set).")
        _model = model_cls(device=DEVICE)
    if INFERENCE_MODE:
        _model.optimize_for_inference()
    return _model


def _get_rectangle_labels_config(parsed_label_config):
    """
    Find the first RectangleLabels tag bound to Image.
    Supports configs with multiple control tags (e.g. TextArea + RectangleLabels + BrushLabels).
    Returns (from_name, to_name, value, labels) or raises ValueError.
    """
    for from_name, info in parsed_label_config.items():
        if info.get("type") != "RectangleLabels":
            continue
        to_names = info.get("to_name") or []
        inputs = info.get("inputs") or []
        for inp in inputs:
            if inp.get("type") == "Image":
                value = inp.get("value")
                if not value:
                    continue
                labels = info.get("labels") or []
                if len(to_names) < 1:
                    continue
                return from_name, to_names[0], value, labels
    raise ValueError(
        "No RectangleLabels tag bound to Image found in the labeling config. "
        "Add a <RectangleLabels> control with toName pointing to an <Image> tag."
    )


def _parse_interactive_context(context: Dict, img_width: int, img_height: int) -> Dict[str, Any]:
    """
    Parse Label Studio context for interactive pre-annotations.
    Extracts keypoints (with is_positive) and optional focus rectangle from context['result'].
    Returns dict with:
      - positive_points: list of (x_px, y_px)
      - negative_points: list of (x_px, y_px)
      - focus_rect: (x, y, x2, y2) in pixels or None
    """
    out = {"positive_points": [], "negative_points": [], "focus_rect": None}
    if not context or not context.get("result"):
        return out
    first = context["result"][0]
    cw = first.get("original_width") or img_width
    ch = first.get("original_height") or img_height
    for r in context["result"]:
        v = r.get("value") or {}
        rtype = r.get("type", "")
        x_pct = v.get("x", 0)
        y_pct = v.get("y", 0)
        x_px = x_pct * cw / 100.0
        y_px = y_pct * ch / 100.0
        is_positive = r.get("is_positive", True)
        if rtype == "keypointlabels":
            if is_positive:
                out["positive_points"].append((x_px, y_px))
            else:
                out["negative_points"].append((x_px, y_px))
        elif rtype == "rectanglelabels":
            w_pct = v.get("width", 0)
            h_pct = v.get("height", 0)
            x2 = x_px + w_pct * cw / 100.0
            y2 = y_px + h_pct * ch / 100.0
            out["focus_rect"] = (x_px, y_px, x2, y2)
    return out


def _box_contains_point(x_pct: float, y_pct: float, w_pct: float, h_pct: float,
                        img_width: int, img_height: int, px: float, py: float) -> bool:
    """Check if point (px, py) in pixels lies inside the box (percent)."""
    x1 = x_pct * img_width / 100.0
    y1 = y_pct * img_height / 100.0
    x2 = x1 + w_pct * img_width / 100.0
    y2 = y1 + h_pct * img_height / 100.0
    return x1 <= px <= x2 and y1 <= py <= y2


def _box_overlaps_rect(x_pct: float, y_pct: float, w_pct: float, h_pct: float,
                       img_width: int, img_height: int, rx1: float, ry1: float, rx2: float, ry2: float) -> bool:
    """Check if box (percent) overlaps rectangle (pixels)."""
    x1 = x_pct * img_width / 100.0
    y1 = y_pct * img_height / 100.0
    x2 = x1 + w_pct * img_width / 100.0
    y2 = y1 + h_pct * img_height / 100.0
    return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)


def _filter_results_by_context(
    pred: Dict[str, Any],
    ctx_parsed: Dict[str, Any],
    img_width: int,
    img_height: int,
) -> Dict[str, Any]:
    """Filter prediction results by interactive context (keypoints / focus rectangle)."""
    results = pred.get("result") or []
    if not results:
        return pred
    positive = ctx_parsed.get("positive_points") or []
    negative = ctx_parsed.get("negative_points") or []
    focus_rect = ctx_parsed.get("focus_rect")

    filtered = []
    for r in results:
        v = r.get("value") or {}
        x, y = v.get("x", 0), v.get("y", 0)
        w, h = v.get("width", 0), v.get("height", 0)
        if focus_rect:
            if not _box_overlaps_rect(x, y, w, h, img_width, img_height, *focus_rect):
                continue
        if negative:
            if any(_box_contains_point(x, y, w, h, img_width, img_height, nx, ny) for nx, ny in negative):
                continue
        if positive:
            if not any(_box_contains_point(x, y, w, h, img_width, img_height, px, py) for px, py in positive):
                continue
        filtered.append(r)

    if not filtered:
        return {"result": [], "score": 0.0, "model_version": pred.get("model_version", "")}
    avg = sum(r.get("score", 0) for r in filtered) / len(filtered)
    return {"result": filtered, "score": avg, "model_version": pred.get("model_version", "")}


class RFDETRBackend(LabelStudioMLBase):
    """
    Object detection ML backend using RF-DETR with optional custom checkpoint.
    Supports interactive pre-annotations: use KeypointLabels (and is_positive) or
    a RectangleLabels region in the labeler to filter predictions by click/region.

    Environment variables:
        CHECKPOINT_PATH: Path to your custom .pt/.pth checkpoint (optional).
        MODEL_SIZE: One of nano, small, medium, large (default: medium).
        CLASSES_FILE: Path to JSON array or newline-separated class names (optional; default COCO).
        SCORE_THRESHOLD: Confidence threshold 0–1 (default: 0.5).
        DEVICE: cpu or cuda (default: cuda if available).
    """

    def __init__(self, image_dir=None, **kwargs):
        super().__init__(**kwargs)
        upload_dir = os.path.join(get_data_dir(), "media", "upload")
        self.image_dir = image_dir or upload_dir
        params = _get_rectangle_labels_config(self.parsed_label_config)
        self.from_name, self.to_name, self.value, self.labels_in_config = params
        self.labels_in_config = set(self.labels_in_config)
        self.score_threshold = SCORE_THRESHOLD
        _load_class_names()

    def setup(self):
        self.set("model_version", f"{self.__class__.__name__}-v0.0.1")

    def _get_image_url(self, task: Dict) -> str:
        return (
            task["data"].get(self.value)
            or task["data"].get(DATA_UNDEFINED_NAME)
            or ""
        )

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """
        Make predictions for the tasks. Supports interactive pre-annotations via context.

        - tasks: Label Studio tasks in JSON format.
        - context: Optional. When present, may contain annotation actions from the labeler:
          - result: list of result items. KeypointLabels (e.g. user clicks) carry is_positive
            (true = positive click, false = e.g. Alt+click for negative). RectangleLabels
            can be used as a focus region. Predictions are filtered to boxes that contain
            a positive keypoint, exclude negative keypoints, and optionally overlap the
            focus rectangle.
        """
        if len(tasks) > 1:
            logger.info("Processing only the first task to avoid overloading.")
            tasks = tasks[:1]
        predictions = []
        for task in tasks:
            pred = self._predict_one(task)
            if context and pred.get("result") and context.get("result"):
                first_ctx = context["result"][0]
                img_width = first_ctx.get("original_width") or 1
                img_height = first_ctx.get("original_height") or 1
                if img_width <= 1 or img_height <= 1:
                    try:
                        image_url = self._get_image_url(task)
                        if image_url:
                            path = get_local_path(image_url, task_id=task.get("id"))
                            img_width, img_height = get_image_size(path)
                    except Exception:
                        pass
                ctx_parsed = _parse_interactive_context(
                    context, img_width, img_height
                )
                has_hint = (
                    ctx_parsed.get("positive_points")
                    or ctx_parsed.get("negative_points")
                    or ctx_parsed.get("focus_rect")
                )
                if has_hint:
                    pred = _filter_results_by_context(
                        pred, ctx_parsed, img_width, img_height
                    )
            predictions.append(pred)
        return predictions

    def _predict_one(self, task: Dict) -> Dict[str, Any]:
        image_url = self._get_image_url(task)
        if not image_url:
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}
        try:
            image_path = get_local_path(image_url, task_id=task.get("id"))
        except Exception as e:
            logger.exception("Failed to get image path: %s", e)
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}
        if not os.path.isfile(image_path):
            logger.warning("Image file not found: %s", image_path)
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        from PIL import Image
        model = _load_model()
        class_names = _load_class_names()
        img_width, img_height = get_image_size(image_path)
        image = Image.open(image_path).convert("RGB")

        detections = model.predict(image, threshold=self.score_threshold)

        results = []
        scores_used = []
        xyxy = getattr(detections, "xyxy", None)
        if xyxy is None and hasattr(detections, "boxes"):
            xyxy = detections.boxes
        if xyxy is None:
            logger.warning("No bounding boxes in detections object: %s", type(detections))
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}

        class_ids = getattr(detections, "class_id", None)
        if class_ids is None and hasattr(detections, "labels"):
            class_ids = detections.labels
        confidences = getattr(detections, "confidence", None)
        if confidences is None and hasattr(detections, "scores"):
            confidences = detections.scores

        import numpy as np
        xyxy = np.asarray(xyxy)
        if xyxy.ndim == 1:
            xyxy = xyxy.reshape(1, -1)
        n = len(xyxy)
        if n == 0:
            return {"result": [], "score": 0.0, "model_version": self.get("model_version")}
        if class_ids is None:
            class_ids = np.zeros(n, dtype=np.int64)
        else:
            class_ids = np.asarray(class_ids).ravel()
        if confidences is None:
            confidences = np.ones(n, dtype=np.float32)
        else:
            confidences = np.asarray(confidences).ravel()

        for i in range(n):
            x1, y1, x2, y2 = xyxy[i][:4]
            cls_id = int(class_ids[i]) if class_ids.size > 0 else 0
            score = float(confidences[i]) if confidences.size > 0 else 1.0
            if cls_id < 0 or cls_id >= len(class_names):
                continue
            label = class_names[cls_id]
            if label not in self.labels_in_config:
                continue
            x_pct = float(x1) / img_width * 100
            y_pct = float(y1) / img_height * 100
            w_pct = (float(x2) - float(x1)) / img_width * 100
            h_pct = (float(y2) - float(y1)) / img_height * 100
            results.append({
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [label],
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                },
                "score": score,
            })
            scores_used.append(score)

        avg_score = sum(scores_used) / len(scores_used) if scores_used else 0.0
        return {
            "result": results,
            "score": avg_score,
            "model_version": self.get("model_version"),
        }
