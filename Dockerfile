FROM python:3.10-slim

WORKDIR /code

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (needed for ONNX export step)
RUN pip install --no-cache-dir --default-timeout=1000 torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./models /code/models

# Persist sync=False in Ultralytics settings YAML BEFORE any model operations.
# This ensures the Events class reads sync=False at import time (not after).
RUN python -c "from ultralytics import settings; settings.update({'sync': False})"

# Export model to ONNX for faster CPU inference (~30-50% speedup over PyTorch)
RUN python -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='onnx', imgsz=640, simplify=True)"

# B3 SKU: 4 vCPU, 7 GB RAM, 10 GB storage, up to 3 scale instances
# Running 2 instances for redundancy and stability
# Optimal threading: 2 threads per worker × 2 workers = 4 total = B3 core count per instance
ENV OMP_NUM_THREADS=2
ENV ORT_NUM_THREADS=2
ENV MODEL_PATH=models/best.onnx
ENV YOLO_IMG_SIZE=640
ENV YOLO_VERBOSE=false

EXPOSE 80

# 2 workers per instance: each gets 2 CPU cores worth of threads for faster per-request inference
# With 2 instances × 2 workers = 4 total workers across the service for high availability
CMD ["gunicorn", "app.main:app", "--bind", "0.0.0.0:80", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "300"]
