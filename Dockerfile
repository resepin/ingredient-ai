# 1. Use Python 3.10
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /code

# 3. Install System Libraries (Keep this, it's perfect for OpenCV)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 4. Install CPU-only PyTorch (Keep this, it saves massive space)
RUN pip install --no-cache-dir --default-timeout=1000 torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Copy application code
COPY ./app /code/app
COPY ./models /code/models

# 7. Expose Port 80
EXPOSE 80

# 8. STARTUP COMMAND (OPTIMIZED)
# -k uvicorn.workers.UvicornWorker: Tells Gunicorn to use Uvicorn for speed
# -w 2: Spawns 2 worker processes (Best for B1 plan with 1.75GB RAM)
# --timeout 600: Prevents "Critical Worker Timeout" if YOLO takes too long
CMD ["gunicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "600"]