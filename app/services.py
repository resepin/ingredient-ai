import gc
import io
import logging
import os

from dotenv import load_dotenv
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))

logger.info("Loading YOLO model from: %s", MODEL_PATH)
model = YOLO(MODEL_PATH)

try:
    model.fuse()
    logger.info("Model fused for faster inference.")
except Exception:
    logger.info("Model fusion not available, using standard model.")

logger.info("Model loaded. Inference image size: %dpx", IMG_SIZE)


def predict_food_items(image_bytes: bytes) -> list[str]:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()

        img = Image.open(io.BytesIO(image_bytes))

        results = model.predict(
            source=img,
            save=False,
            imgsz=IMG_SIZE,
            verbose=False,
        )

        detected_items: set[str] = set()
        for r in results:
            if hasattr(r.boxes, "cls"):
                for c in r.boxes.cls:
                    detected_items.add(model.names[int(c)])  # type: ignore

        del img, results
        gc.collect()

        return list(detected_items)

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return []
