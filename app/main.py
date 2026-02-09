import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router
from app.services import model, IMG_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Application Insights telemetry
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(
            enable_live_metrics=True,
        )
        print("✅ Application Insights telemetry initialized (Live Metrics enabled)")
        logger.info("Application Insights telemetry initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Application Insights: {e}")
        logger.error("Failed to initialize Application Insights: %s", e)
else:
    print("⚠️  APPLICATIONINSIGHTS_CONNECTION_STRING not set - telemetry disabled")
    logger.warning("APPLICATIONINSIGHTS_CONNECTION_STRING not set - telemetry disabled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the YOLO model on startup so the first real request is fast."""
    try:
        import numpy as np
        from PIL import Image
        print("Warming up YOLO model...")
        logger.info("Warming up model...")
        dummy_img = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        model.predict(dummy_img, imgsz=IMG_SIZE, verbose=False)
        print("✅ Model warmup complete - ready for inference")
        logger.info("Model warmup complete.")
    except Exception as e:
        print(f"⚠️  Model warmup failed: {e}")
        logger.warning("Model warmup failed: %s", e)
    yield


app = FastAPI(
    title="Ingredient Detection API",
    description="YOLOv8-powered food ingredient detection service optimized for B3",
    version="1.3.0",
    lifespan=lifespan,
)

default_origins = "http://localhost:8000,http://127.0.0.1:8000,https://resepin.azurewebsites.net"
raw_origins = os.getenv("ALLOWED_ORIGINS", default_origins)
origins = [origin.strip() for origin in raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Service Unhealthy: Model not loaded")
    return {"status": "Healthy", "version": app.version}
