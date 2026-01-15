import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
from app.services import model

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Ingredient Detection API",
    description="An API that uses YOLOv8 to detect food ingredients for Laravel",
    version="1.0.1"
)

# --- Dynamic CORS Configuration ---
# Get the origins from .env or Azure settings. 
# Default to localhost if nothing is found.
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")

origins = [origin.strip() for origin in raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Include the routes from routes.py
# This connects our /predict endpoint to the main app
app.include_router(router)

# 3. Root endpoint for health checks
@app.get("/")
def read_root():
    if model is None:
        raise HTTPException(status_code=503, detail="Service Unhealthy: Model not loaded")
    return {"status": "Healthy", "message": "AI Service is running!"}