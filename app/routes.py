from fastapi import APIRouter, File, UploadFile
from app.services import predict_food_items
from app.schemas import DetectionResponse
from fastapi.concurrency import run_in_threadpool

router = APIRouter()

# Add response_model=DetectionResponse
@router.post("/predict", response_model=DetectionResponse)
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    # threading for performance
    detected_ingredients = await run_in_threadpool(predict_food_items, contents)
    
    # Return matches the Schema structure (object with 'ingredients' list)
    return {"ingredients": detected_ingredients}