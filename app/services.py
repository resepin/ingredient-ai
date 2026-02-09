import io
import logging
import os
import time

from dotenv import load_dotenv
from fastapi import HTTPException
from opentelemetry import metrics
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

load_dotenv()

logger = logging.getLogger(__name__)

# Custom metric: tracks pure inference time (what you see in Postman locally)
meter = metrics.get_meter("resepin-api")
inference_histogram = meter.create_histogram(
    name="inference_duration_ms",
    description="Pure YOLO model inference time in milliseconds",
    unit="ms",
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

        # Measure pure inference time (excluding network, I/O, serialization)
        inference_start = time.perf_counter()
        results = model.predict(
            source=img,
            save=False,
            imgsz=IMG_SIZE,
            verbose=False,
        )
        inference_duration_ms = (time.perf_counter() - inference_start) * 1000

        detected_items: set[str] = set()
        if results:
            r = results[0]
            if hasattr(r.boxes, "cls") and len(r.boxes.cls) > 0:
                for c in r.boxes.cls:
                    detected_items.add(model.names[int(c)])  # type: ignore

        # Send pure inference time to Application Insights as a custom metric
        inference_histogram.record(inference_duration_ms)
        logger.info("Inference completed in %.1fms", inference_duration_ms)

        return list(detected_items)

    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return []
