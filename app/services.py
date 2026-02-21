import io
import logging
import os
import time

from dotenv import load_dotenv
from fastapi import HTTPException
from opentelemetry import metrics
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO, settings as yolo_settings
from ultralytics.hub.utils import events as _ul_events

load_dotenv()

logger = logging.getLogger(__name__)

# Disable Ultralytics analytics (Google Analytics calls).
# settings.update() writes to YAML but the Events instance already captured
# enabled=True at import time, so we also kill it directly.
yolo_settings.update({"sync": False})
_ul_events.enabled = False

# Custom metrics: tracks inference time for P50/P90/P99 percentile monitoring
meter = metrics.get_meter("resepin-api")
inference_histogram = meter.create_histogram(
    name="inference_duration_ms",
    description="Pure YOLO model inference time in milliseconds (P50/P90/P99)",
    unit="ms",
)
inference_counter = meter.create_counter(
    name="inference_count",
    description="Total number of inference requests processed",
    unit="1",
)
inference_error_counter = meter.create_counter(
    name="inference_error_count",
    description="Total number of failed inference requests",
    unit="1",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.onnx")
IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))

print(f"Loading YOLO model from: {MODEL_PATH}")
logger.info("Loading YOLO model from: %s", MODEL_PATH)
model = YOLO(MODEL_PATH)

print(f"✅ Model loaded successfully. Inference image size: {IMG_SIZE}px")
logger.info("Model loaded. Inference image size: %dpx", IMG_SIZE)


def predict_food_items(image_bytes: bytes) -> list[str]:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Pre-resize large images to reduce YOLO internal preprocessing overhead
        max_dim = max(img.size)
        if max_dim > IMG_SIZE:
            img.thumbnail((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        # Measure pure inference time (excluding network, I/O, serialization)
        inference_start = time.perf_counter()
        results = model.predict(
            source=img,
            save=False,
            imgsz=IMG_SIZE,
            verbose=False,
            conf=0.25,
            max_det=50,
        )
        inference_duration_ms = (time.perf_counter() - inference_start) * 1000

        detected_items: set[str] = set()
        if results:
            r = results[0]
            if hasattr(r.boxes, "cls") and len(r.boxes.cls) > 0:
                for c in r.boxes.cls:
                    detected_items.add(model.names[int(c)])  # type: ignore

        # Record metrics for Application Insights (P50/P90/P99 via histogram)
        inference_histogram.record(
            inference_duration_ms,
            {"endpoint": "/predict", "method": "POST"},
        )
        inference_counter.add(
            1,
            {"endpoint": "/predict", "status": "success", "items_detected": str(len(detected_items))},
        )
        logger.info("Inference completed in %.1fms, detected %d items", inference_duration_ms, len(detected_items))

        return list(detected_items)

    except (UnidentifiedImageError, OSError):
        inference_error_counter.add(1, {"endpoint": "/predict", "error": "invalid_image"})
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        inference_error_counter.add(1, {"endpoint": "/predict", "error": type(e).__name__})
        logger.error("Prediction failed: %s", e)
        return []
