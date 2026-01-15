import io
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException
from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
model = YOLO(MODEL_PATH)

def predict_food_items(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        img.verify()
        # verify() consumes the file pointer
        img = Image.open(io.BytesIO(image_bytes))
        
        results = model.predict(source=img, save=False)
        
        detected_items = []
        for r in results:
            for c in r.boxes.cls:
                detected_items.append(model.names[int(c)]) # type: ignore
                
        return list(set(detected_items))

    except UnidentifiedImageError:
        # If PIL cannot open the file
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        # Catch generic model errors
        print(f"Model Error: {e}")
        return [] # Return empty list instead of crashing