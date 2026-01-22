import io
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException
from ultralytics import YOLO
from dotenv import load_dotenv
import os
import gc 

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")

# Load model globally so it stays in memory (Fastest Inference)
print(f"Loading YOLO Model from: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

def predict_food_items(image_bytes: bytes):
    try:
        # 1. Verify Image
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        
        # 2. Re-open for processing
        img = Image.open(io.BytesIO(image_bytes))
        
        # 3. Run Inference
        # conf=0.25: Only return high-confidence items to reduce noise
        results = model.predict(source=img, save=False, conf=0.25)
        
        detected_items = []
        for r in results:
            if hasattr(r.boxes, 'cls'):
                for c in r.boxes.cls:
                    class_name = model.names[int(c)] # type: ignore
                    detected_items.append(class_name)
        
        # Force garbage collection to keep RAM usage low on B1 Plan
        del img
        del results
        gc.collect()

        return list(set(detected_items))

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        print(f"Model Error: {e}")
        return []