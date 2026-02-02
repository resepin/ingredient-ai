import io
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException
from ultralytics import YOLO
from dotenv import load_dotenv
import os
import gc 

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")

# Optimal image size for B1 - balances speed and accuracy
# 480px maintains good accuracy while being faster than 640px
# Works on B1 and scales up automatically with better plans
IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "480"))

# Load model globally so it stays in memory (Fastest Inference)
print(f"Loading YOLO Model from: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# Optimize model for faster inference
try:
    model.fuse()  # Fuse Conv2d + BatchNorm layers for 10-15% speedup
    print("Model fused for faster inference")
except:
    print("Model fusion not available, continuing with standard model")

print(f"Model loaded successfully! Using image size: {IMG_SIZE}px")

def predict_food_items(image_bytes: bytes):
    try:
        # 1. Verify Image
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        
        # 2. Re-open for processing
        img = Image.open(io.BytesIO(image_bytes))
        
        # 3. Run Inference with optimized settings
        # imgsz=480: Faster than 640 while maintaining accuracy on B1
        # conf=0.25: Only return high-confidence items to reduce noise
        # verbose=False: Reduce console spam
        results = model.predict(
            source=img, 
            save=False, 
            # conf=0.25,
            imgsz=IMG_SIZE,
            verbose=False
        )
        
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