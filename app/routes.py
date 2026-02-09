from fastapi import APIRouter, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas import DetectionResponse
from app.services import predict_food_items

router = APIRouter()


@router.post("/predict", response_model=DetectionResponse)
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    detected_ingredients = await run_in_threadpool(predict_food_items, contents)
    return {"ingredients": detected_ingredients}
