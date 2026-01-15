from pydantic import BaseModel
from typing import List

# This defines the "shape" of the JSON response
# It guarantees that Laravel always receives a list of strings
class DetectionResponse(BaseModel):
    ingredients: List[str]