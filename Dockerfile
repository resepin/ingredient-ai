# 1. Use Python 3.10
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /code

# 3. Install System Libraries for OpenCV and YOLO
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 4. Install CPU-only PyTorch first (Optimized for Azure Web Apps)
RUN pip install --no-cache-dir --default-timeout=1000 torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Copy and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 --upgrade -r /code/requirements.txt

# 6. Copy application code and models
COPY ./app /code/app
COPY ./models /code/models

# 7. Expose Port 80 (standard for Azure Web Apps)
EXPOSE 80

# 8. Start the server
# We use app.main:app because your structure is /code/app/main.py
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]