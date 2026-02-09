from pydantic import BaseModel

class DetectionResponse(BaseModel):
    ingredients: list[str]
