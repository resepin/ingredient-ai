FROM python:3.10-slim

WORKDIR /code

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (needed for ONNX export step)
RUN pip install --no-cache-dir --default-timeout=1000 torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./models /code/models

# Export model to ONNX for faster CPU inference (~30-50% speedup over PyTorch)
RUN python -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='onnx', imgsz=640, simplify=True)"

# Optimal threading: 2 threads per worker × 2 workers = 4 total = B3 core count
ENV OMP_NUM_THREADS=2
ENV MODEL_PATH=models/best.onnx
ENV YOLO_IMG_SIZE=640

EXPOSE 80

# 2 workers: each gets 2 CPU cores worth of threads for faster per-request inference
# (4 workers would split cores and slow each request down)
CMD ["gunicorn", "app.main:app", "--bind", "0.0.0.0:80", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "300"]
